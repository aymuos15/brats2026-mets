"""lbquery2 — pull ONE snapshot of the scored leaderboard to CSV.

Rebuilt 2026-07-17 (the original was lost; see handoff §0b).
Run:  PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring \
      ~/brats2025/.venv/bin/python lbquery2.py [out.csv]

Every mean rank is only comparable WITHIN one snapshot (handoff §0), so this
writes a timestamped file and rank12.py consumes exactly one of them. Never
mix two. venv has no pandas -> csv module only.
"""
import os, sys, csv, shutil, synapseclient

TABLE = "syn74508245"
HOME = os.path.expanduser("~")
out = sys.argv[1] if len(sys.argv) > 1 else f"{HOME}/brats2025/lb_snapshot.csv"

tok = open(f"{HOME}/brats2025/syn.tok").read().strip()
syn = synapseclient.Synapse(silent=True)
syn.login(authToken=tok)

# SCORED only — anything else has no metrics and would pollute the ranks.
q = syn.tableQuery(f"SELECT * FROM {TABLE} WHERE submission_status = 'SCORED'")
shutil.copy(q.filepath, out)

with open(out) as f:
    n = sum(1 for _ in csv.reader(f)) - 1
print(f"[lbquery2] wrote {out}  ({n} scored submissions)")
print("LBQUERY_COMPLETE")
