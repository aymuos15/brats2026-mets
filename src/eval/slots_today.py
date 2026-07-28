"""Slots used today — ALL statuses, not just SCORED.

lbquery2.py filters submission_status='SCORED', so a submission made in the last
hour (still SCORING/RECEIVED/VALIDATED) is invisible to it. Slots are consumed at
SUBMIT time, not at score time -- so this must look at every status.
"""
import os, csv, datetime, synapseclient

OURS = "3495063"
HOME = os.path.expanduser("~")
tok = open(f"{HOME}/brats2025/syn.tok").read().strip()
syn = synapseclient.Synapse(silent=True)
syn.login(authToken=tok)

q = syn.tableQuery(f"SELECT * FROM syn74508245 WHERE submitterid = '{OURS}'")
rows = list(csv.DictReader(open(q.filepath)))


def bst(r):
    return datetime.datetime.utcfromtimestamp(int(r["createdOn"]) / 1000) + datetime.timedelta(hours=1)


rows.sort(key=lambda r: int(r["createdOn"]))
print(f"{len(rows)} total rows for team {OURS} (ALL statuses)\n")
print("--- last 8, any status ---")
print(f"{'id':>9} {'created BST':>13} {'status':>10} {'sub_status':>12}  name")
for r in rows[-8:]:
    print(f"{r['id']:>9} {bst(r).strftime('%m-%d %H:%M'):>13} {r['status']:>10} "
          f"{r['submission_status']:>12}  {r['name'][:44]}")

print()
for day in ["2026-07-15", "2026-07-16", "2026-07-17"]:
    d = datetime.date.fromisoformat(day)
    n = [r for r in rows if bst(r).date() == d]
    tag = "  <-- TODAY" if day == "2026-07-17" else ""
    print(f"  {day}: {len(n)} submission(s) of any status "
          f"{[(r['id'], r['submission_status']) for r in n]}{tag}")

print("\nNOTE: a slot is consumed at SUBMIT time regardless of whether it has scored yet.")
print("NOTE: this table may only expose rows the scoring harness has ingested; a submission")
print("      rejected before ingest (bad zip) could still have burned a slot and not appear.")
