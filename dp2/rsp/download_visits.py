"""
Download the DP2 visit metadata, to define sky fields from the observing programme
instead of from object positions.

Runs on the Rubin Science Platform, cell by cell in a notebook — `lsst.rsp` exists
only there.

After running: download `dp2_visits.csv` and `dp2_visit_schema.csv` into
`dp2/catalogs/` on the laptop.
"""

# %% ── 1. Setup ───────────────────────────────────────────────────────────────
import pandas as pd
from lsst.rsp import get_tap_service

service = get_tap_service("tap")
assert service is not None

VISIT_TABLE = "dp2.Visit"


def query(adql):
    job = service.submit_job(adql)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    job.raise_if_error()
    return job.fetch_result().to_table().to_pandas()


# %% ── 2. Schema: which columns describe the pointing? ─────────────────────────
# Look for a target / field / programme name (e.g. target_name, science_program,
# observation_reason) and the pointing centre (ra, dec / boresight).
schema = query(f"""
    SELECT column_name, datatype, unit, description
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{VISIT_TABLE}'
    ORDER BY column_name
""")
print(schema.to_string(index=False))
schema.to_csv("dp2_visit_schema.csv", index=False)


# %% ── 3. Every visit, every column ───────────────────────────────────────────
# One row per visit: a few tens of thousands of rows at most, small enough to take
# whole rather than guessing which columns matter.
visits = query(f"SELECT * FROM {VISIT_TABLE}")
print(f"{len(visits):,} visits x {visits.shape[1]} columns")
visits.to_csv("dp2_visits.csv", index=False)
