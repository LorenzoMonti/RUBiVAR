"""
Download DP2 light curves for the objects in the labelled catalog.

Runs on the Rubin Science Platform, cell by cell in a notebook — `lsst.rsp` exists
only there. Everything else in this project runs locally.

Before running: upload `catalogs/dp2_labeled_catalog.csv` to the RSP home directory.
After running : download `lightcurves/` back into `dp2/lightcurves/` on the laptop.

The query is resumable: rerun the download cell as many times as needed and it picks
up where it stopped.
"""

# %% ── 1. Setup ───────────────────────────────────────────────────────────────
import os
import time

import pandas as pd
from lsst.rsp import get_tap_service

service = get_tap_service("tap")
assert service is not None

CATALOG_FILE = "dp2_labeled_catalog.csv"   # uploaded from catalogs/
OUTPUT_DIR   = "lightcurves"
# Objects per IN-clause query. Deliberately smaller than for DiaSource: forced
# photometry returns a row per visit per band, so a batch is a few hundred thousand
# rows. Raise it if the queries come back comfortably fast.
BATCH_SIZE   = 100

# Per-epoch photometry, keyed on diaObjectId.
#
# ForcedSourceOnDiaObject holds forced photometry at EVERY visit, so a measurement
# exists even when the star is at minimum — no selection against the faint half of
# the cycle, which is what we want for phase-folding.
# DiaSource would give only difference-image detections, i.e. the epochs where the
# object was significantly variable (that is what DP1 used).
# DiaObject is not an option: per-band aggregates, one row per object, no time axis.
SOURCE_TABLE = "dp2.ForcedSourceOnDiaObject"

# Visit metadata, joined in when the per-epoch table carries no timestamp of its own.
VISIT_TABLE = "dp2.Visit"


# %% ── 1a. Which tables exist? ────────────────────────────────────────────────
tables_job = service.submit_job("""
    SELECT table_name FROM TAP_SCHEMA.tables
    WHERE table_name LIKE 'dp2.%'
    ORDER BY table_name
""")
tables_job.run()
tables_job.wait(phases=["COMPLETED", "ERROR"])
print(tables_job.fetch_result().to_table().to_pandas().to_string(index=False))

# Candidates for per-epoch photometry, in order of preference:
#   ForcedSourceOnDiaObject / ForcedSource — forced photometry at every visit,
#       so a measurement exists even when the object is faint. Better for
#       phase-folding: no selection against the faint half of the cycle.
#   DiaSource — difference-image detections only, i.e. epochs where the object
#       was significantly variable. This is what DP1 used.


# %% ── 1b. Build the query from the real schema ─────────────────────────────────
# Nothing about the column names is hardcoded: ask the schema and use what is there.
# ForcedSourceOnDiaObject usually carries no timestamp of its own — the epoch time
# lives in the Visit table and has to be joined in on the visit key.

TIME_CANDIDATES  = ["expMidptMJD", "midpointMjdTai", "midPointTai", "midpointTai",
                    "obsStartMJD", "expMidpt"]
VISIT_KEYS       = ["visit", "visitId", "ccdVisitId"]
# Needed by lightcurves.py; a missing one is fatal.
REQUIRED         = ["diaObjectId", "band", "psfFlux", "psfFluxErr"]
# Kept when present, skipped without complaint when not.
# DP1 kept visit + coord_ra/coord_dec alongside both fluxes; same list here, plus
# the _flag columns that DP1 did not select.
OPTIONAL         = ["visit", "psfDiffFlux", "psfDiffFluxErr",
                    "scienceFlux", "scienceFluxErr",
                    "coord_ra", "coord_dec", "ra", "dec", "detector",
                    "psfFlux_flag", "psfDiffFlux_flag", "snr"]


def fetch_schema(table):
    job = service.submit_job(f"""
        SELECT column_name, datatype, description FROM TAP_SCHEMA.columns
        WHERE table_name = '{table}'
    """)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    if job.phase == "ERROR":
        job.raise_if_error()
    return job.fetch_result().to_table().to_pandas()


src_schema = fetch_schema(SOURCE_TABLE)
src_cols = set(src_schema["column_name"])
print(f"{SOURCE_TABLE}: {len(src_cols)} columns\n")

# Eyeball these the first time. A per-epoch table shows a bare psfFlux; per-band
# aggregates (g_psfFluxMean, r_psfFluxSigma, nDiaSources) mean you are on DiaObject.
relevant = src_schema[src_schema["column_name"].str.contains(
    r"^(?:psf|science)?(?:Diff)?[Ff]lux|[Mm]jd|[Tt]ai|[Mm]id[Pp]oint|band|^coord_|"
    r"^ra$|^dec$|visit|snr|diaObjectId", regex=True)]
print(relevant[["column_name", "datatype", "description"]].to_string(index=False))
print()

missing = [c for c in REQUIRED if c not in src_cols]
if missing:
    raise KeyError(f"{SOURCE_TABLE} has no column(s) {missing} — check the schema above.")

select = [f"s.{c}" for c in REQUIRED] + [f"s.{c}" for c in OPTIONAL if c in src_cols]

# ── Time: from the source table if it has one, otherwise joined from Visit ────
TIME_COL = next((c for c in TIME_CANDIDATES if c in src_cols), None)
if TIME_COL:
    from_clause = f"{SOURCE_TABLE} AS s"
    select.append(f"s.{TIME_COL}")
    print(f"Time column : s.{TIME_COL}  (no join needed)")
else:
    visit_schema = fetch_schema(VISIT_TABLE)
    visit_cols = set(visit_schema["column_name"])
    TIME_COL = next((c for c in TIME_CANDIDATES if c in visit_cols), None)
    JOIN_KEY = next((k for k in VISIT_KEYS if k in src_cols and k in visit_cols), None)
    if TIME_COL is None or JOIN_KEY is None:
        print(visit_schema[["column_name", "datatype"]].to_string(index=False))
        raise KeyError(
            f"Cannot reach a timestamp: time={TIME_COL}, join key={JOIN_KEY}. "
            f"Pick them from the {VISIT_TABLE} schema printed above."
        )
    from_clause = (f"{SOURCE_TABLE} AS s "
                   f"JOIN {VISIT_TABLE} AS v ON s.{JOIN_KEY} = v.{JOIN_KEY}")
    select.append(f"v.{TIME_COL}")
    print(f"Time column : v.{TIME_COL}  (joined on {JOIN_KEY})")

SELECT_LIST = ", ".join(select)
FROM_CLAUSE = from_clause
TIME_REF    = select[-1]          # qualified time column, for ORDER BY
print(f"Selecting   : {SELECT_LIST}")
print(f"From        : {FROM_CLAUSE}")


# %% ── 2. Target list ──────────────────────────────────────────────────────────
targets = pd.read_csv(CATALOG_FILE, dtype={"diaObjectId": str})
print(f"Objects in catalog: {len(targets):,}")
print(targets["label"].value_counts().to_string())

# Highest variability significance first, so an interrupted run still leaves a
# usable, well-ordered subset behind.
targets = targets.sort_values("r_var_significance", ascending=False)
target_ids = targets["diaObjectId"].tolist()


# %% ── 3. Download (resumable) ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

done = {f[3:-4] for f in os.listdir(OUTPUT_DIR)
        if f.startswith("lc_") and f.endswith(".csv")}
todo = [i for i in target_ids if i not in done]
print(f"Already downloaded: {len(done):,}")
print(f"Remaining         : {len(todo):,}")

t0 = time.time()
for start in range(0, len(todo), BATCH_SIZE):
    batch = todo[start:start + BATCH_SIZE]
    id_list = ", ".join(batch)
    query = f"""
        SELECT {SELECT_LIST}
        FROM {FROM_CLAUSE}
        WHERE s.diaObjectId IN ({id_list})
        ORDER BY s.diaObjectId, {TIME_REF}
    """
    try:
        job = service.submit_job(query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR"])
        if job.phase == "ERROR":
            job.raise_if_error()
        df = job.fetch_result().to_table().to_pandas()
    except Exception as exc:
        print(f"  batch at {start} failed: {exc}")
        continue

    df["diaObjectId"] = df["diaObjectId"].astype(str)
    for oid, sub in df.groupby("diaObjectId"):
        sub.to_csv(os.path.join(OUTPUT_DIR, f"lc_{oid}.csv"), index=False)

    # Objects the query returned nothing for: write an empty marker so the resume
    # logic does not retry them on every rerun.
    for oid in set(batch) - set(df["diaObjectId"]):
        pd.DataFrame(columns=df.columns).to_csv(
            os.path.join(OUTPUT_DIR, f"lc_{oid}.csv"), index=False)

    n_done = start + len(batch)
    rate = n_done / max(time.time() - t0, 1e-6)
    eta = (len(todo) - n_done) / max(rate, 1e-9) / 60
    print(f"  {n_done:,}/{len(todo):,}  ({rate:.1f} obj/s, ETA {eta:.0f} min)")

print(f"\nTotal files: {len(os.listdir(OUTPUT_DIR)):,}")


# %% ── 4. Sanity check on epoch counts ─────────────────────────────────────────
# The number of measurements per object sets the fold quality; this is the
# distribution the local `main.py fold --min-points` threshold should be based on.
counts = []
for f in os.listdir(OUTPUT_DIR):
    if f.startswith("lc_") and f.endswith(".csv"):
        counts.append(sum(1 for _ in open(os.path.join(OUTPUT_DIR, f))) - 1)

counts = pd.Series(counts)
print(f"Light curves : {len(counts):,}")
print(f"Empty        : {int((counts <= 0).sum()):,}")
print("\nMeasurements per object:")
print(counts[counts > 0].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())

# %% ── 5. Zip the output files ─────────────────────────────────────────
import shutil

# Percorso della cartella da comprimere
cartella_da_zippare = 'lightcurves_dp2'

# Nome del file zip da creare (senza estensione .zip)
nome_archivio = 'lightcurves_dp2'

# Crea il file zip
shutil.make_archive(nome_archivio, 'zip', cartella_da_zippare)