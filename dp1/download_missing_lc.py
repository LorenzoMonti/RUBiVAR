"""
Download light curves for the 130 labeled objects missing from local LC directories.

Run cell-by-cell in a Jupyter notebook on the Rubin Science Platform.

Inputs  : missing_lc_ids.csv  (upload this file to the RSP home directory)
Outputs : lc_xmatch/lc_{diaObjectId}.csv  — one file per object
"""

# %% ── 1. Imports ─────────────────────────────────────────────────────────────
import os
import pandas as pd
from lsst.rsp import get_tap_service

# %% ── 2. Config ──────────────────────────────────────────────────────────────
IDS_FILE   = "missing_lc_ids.csv"   # upload this to RSP alongside the notebook
OUTPUT_DIR = "lc_xmatch"
BATCH_SIZE = 50                      # objects per IN-clause query

# %% ── 3. Load target IDs ─────────────────────────────────────────────────────
service = get_tap_service("tap")
assert service is not None

missing = pd.read_csv(IDS_FILE)
print(f"Target objects: {len(missing)}")
print(missing["label"].value_counts().to_string())

# %% ── 4. Download (auto-resume) ──────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

already_done = {
    int(f[3:-4])
    for f in os.listdir(OUTPUT_DIR)
    if f.startswith("lc_") and f.endswith(".csv")
}
print(f"Already downloaded: {len(already_done)}")

todo = missing[~missing["diaObjectId"].isin(already_done)].reset_index(drop=True)
print(f"Remaining         : {len(todo)}")

batches = [todo.iloc[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
print(f"Batches           : {len(batches)}  (x{BATCH_SIZE} objects each)\n")

failed = []

for b_idx, batch in enumerate(batches):
    ids_str = ", ".join(str(i) for i in batch["diaObjectId"])
    print(f"[{b_idx + 1}/{len(batches)}]", end=" ", flush=True)

    query = f"""
    SELECT
        fsodo.diaObjectId,
        fsodo.coord_ra,
        fsodo.coord_dec,
        fsodo.visit,
        fsodo.band,
        fsodo.psfFlux,
        fsodo.psfFluxErr,
        fsodo.psfDiffFlux,
        fsodo.psfDiffFluxErr,
        vis.expMidptMJD
    FROM dp1.ForcedSourceOnDiaObject AS fsodo
    JOIN dp1.Visit AS vis ON vis.visit = fsodo.visit
    WHERE fsodo.diaObjectId IN ({ids_str})
    ORDER BY fsodo.diaObjectId, vis.expMidptMJD
    """

    try:
        job = service.submit_job(query)
        job.run()
        job.wait()
        job.raise_if_error()

        batch_lc = job.fetch_result().to_table().to_pandas()
        job.delete()

        if len(batch_lc) == 0:
            print("no data.")
            continue

        for obj_id, lc in batch_lc.groupby("diaObjectId"):
            lc.to_csv(os.path.join(OUTPUT_DIR, f"lc_{obj_id}.csv"), index=False)

        n_obj = batch_lc["diaObjectId"].nunique()
        n_pts = len(batch_lc)
        print(f"{n_obj} objects, {n_pts} points.")

    except Exception as e:
        print(f"ERROR: {e}")
        failed.extend(batch["diaObjectId"].tolist())

# %% ── 5. Report ──────────────────────────────────────────────────────────────
downloaded = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")])
print(f"\nDone.")
print(f"  Files in {OUTPUT_DIR}/: {downloaded}")
print(f"  Failed batches      : {len(failed)}")

if failed:
    pd.Series(failed, name="diaObjectId").to_csv("failed_ids_xmatch.csv", index=False)
    print("  Failed IDs saved to: failed_ids_xmatch.csv")
