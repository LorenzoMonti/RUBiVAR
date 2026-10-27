"""
Gaia → LSST DP1 positional crossmatch
======================================
Run this script cell-by-cell in a Jupyter notebook on the Rubin Science Platform.

Inputs  : 4 Gaia CSV files in "Gaia catalogues/" (source_id, ra, dec, ...)
Outputs : gaia_lsst_xmatch.csv  — matched catalog with source_id, diaObjectId, label
"""

# %% ── 1. Imports ─────────────────────────────────────────────────────────────
import pandas as pd
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
from lsst.rsp import get_tap_service

# %% ── 2. Config ──────────────────────────────────────────────────────────────
GAIA_DIR          = "Gaia catalogues"
LSST_TABLE        = "dp1.DiaObject"
XMATCH_RADIUS_DEG = 2.0 / 3600.0    # 2 arcsec
FOOTPRINT_MARGIN  = 0.5             # deg of padding around LSST bounding box
CHUNK_SIZE        = 50_000          # rows per TAP upload (stays under 32 MB)
MAX_PARALLEL      = 10              # max concurrent async TAP jobs
OUTPUT_FILE       = "gaia_lsst_xmatch.csv"

CLASS_FILES = {
    "RR"  : "rrlyrae.csv",
    "Ceph": "cepheids.csv",
    "LPV" : "lpv.csv",
    "ECL" : "eb.csv",
}

# %% ── 3. Load and merge Gaia catalogs ────────────────────────────────────────
dfs = []
for label, fname in CLASS_FILES.items():
    df = pd.read_csv(f"{GAIA_DIR}/{fname}", usecols=["source_id", "ra", "dec"])
    df["label"] = label
    dfs.append(df)
    print(f"  {label:4s}: {len(df):6,} sources  ({fname})")

gaia = pd.concat(dfs, ignore_index=True)

priority = {"RR": 0, "Ceph": 1, "LPV": 2, "ECL": 3}
gaia["_pri"] = gaia["label"].map(priority)
gaia = (gaia.sort_values("_pri")
            .drop_duplicates("source_id", keep="first")
            .drop(columns="_pri")
            .reset_index(drop=True))

print(f"\nTotal unique Gaia sources: {len(gaia):,}")
print(gaia["label"].value_counts().to_string())

# %% ── 4. Connect to RSP TAP service ─────────────────────────────────────────
service = get_tap_service("tap")
print("TAP service ready:", service.baseurl)

# %% ── 5. Get LSST DP1 footprint bounding box ────────────────────────────────
print("Fetching LSST footprint bounds …")
bounds_job = service.submit_job(
    f"SELECT MIN(ra) AS ra_min, MAX(ra) AS ra_max, "
    f"MIN(dec) AS dec_min, MAX(dec) AS dec_max "
    f"FROM {LSST_TABLE}"
)
bounds_job.run()
bounds_job.wait()
bounds_job.raise_if_error()
b = bounds_job.fetch_result().to_table().to_pandas().iloc[0]
bounds_job.delete()

ra_min  = b["ra_min"]  - FOOTPRINT_MARGIN
ra_max  = b["ra_max"]  + FOOTPRINT_MARGIN
dec_min = b["dec_min"] - FOOTPRINT_MARGIN
dec_max = b["dec_max"] + FOOTPRINT_MARGIN
print(f"LSST footprint: RA [{ra_min:.2f}, {ra_max:.2f}]  Dec [{dec_min:.2f}, {dec_max:.2f}]")

gaia_f = gaia[
    (gaia["ra"]  >= ra_min) & (gaia["ra"]  <= ra_max) &
    (gaia["dec"] >= dec_min) & (gaia["dec"] <= dec_max)
].reset_index(drop=True)

print(f"\nGaia sources inside footprint: {len(gaia_f):,} / {len(gaia):,}  "
      f"({100 * len(gaia_f) / len(gaia):.1f}%)")
print(gaia_f["label"].value_counts().to_string())

# %% ── 6. Parallel async crossmatch — Gaia → dp1.DiaObject ───────────────────
#
# Each chunk is submitted as an independent async TAP job (no timeout).
# MAX_PARALLEL jobs run concurrently, reducing wall-clock time to ~1 chunk.

query_template = f"""
SELECT
    u.source_id,
    u.label,
    u.ra  AS gaia_ra,
    u.dec AS gaia_dec,
    d.diaObjectId,
    d.ra  AS lsst_ra,
    d.dec AS lsst_dec,
    DISTANCE(
        POINT('ICRS', u.ra, u.dec),
        POINT('ICRS', d.ra, d.dec)
    ) * 3600.0 AS sep_arcsec
FROM TAP_UPLOAD.gaia AS u
JOIN {LSST_TABLE} AS d
  ON 1 = CONTAINS(
        POINT('ICRS', d.ra, d.dec),
        CIRCLE('ICRS', u.ra, u.dec, {XMATCH_RADIUS_DEG})
     )
"""

chunks  = [gaia_f.iloc[i : i + CHUNK_SIZE] for i in range(0, len(gaia_f), CHUNK_SIZE)]
n_total = len(chunks)
print(f"Submitting {n_total} chunk(s) — max {MAX_PARALLEL} concurrent …")


def _run_chunk(idx_chunk):
    idx, chunk = idx_chunk
    upload = Table.from_pandas(chunk[["source_id", "ra", "dec", "label"]])
    job = service.submit_job(query_template, uploads={"gaia": upload})
    job.run()
    job.wait()
    job.raise_if_error()
    df = job.fetch_result().to_table().to_pandas()
    job.delete()
    print(f"  [{idx + 1}/{n_total}] {len(df):,} raw matches", flush=True)
    return df


results = [None] * n_total
with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
    futures = {pool.submit(_run_chunk, (i, c)): i for i, c in enumerate(chunks)}
    for future in as_completed(futures):
        results[futures[future]] = future.result()

xmatch = pd.concat(results, ignore_index=True)
print(f"\nTotal raw matches: {len(xmatch):,}")

# %% ── 7. Keep best match per Gaia source and per LSST object ─────────────────
xmatch = (xmatch
          .sort_values("sep_arcsec")
          .drop_duplicates("source_id", keep="first"))

xmatch = (xmatch
          .sort_values("sep_arcsec")
          .drop_duplicates("diaObjectId", keep="first")
          .sort_values("source_id")
          .reset_index(drop=True))

print(f"Matched pairs after deduplication: {len(xmatch):,}")
print("\nMatches per class:")
print(xmatch["label"].value_counts().to_string())
print(f"\nSeparation stats (arcsec):\n{xmatch['sep_arcsec'].describe().round(4).to_string()}")

# %% ── 8. Save ────────────────────────────────────────────────────────────────
cols_out = ["source_id", "diaObjectId", "label",
            "gaia_ra", "gaia_dec", "lsst_ra", "lsst_dec", "sep_arcsec"]
xmatch[cols_out].to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}")
