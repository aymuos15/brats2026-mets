import csv, datetime, os, sys

OURS = "3495063"
p = os.path.expanduser("~/brats2025/lb_snapshot.csv")
rows = [r for r in csv.DictReader(open(p)) if r["submitterid"] == OURS]


def g(r, m):
    s = (r.get(m) or "").strip()
    return f"{float(s):.3f}"[1:] if s else "  — "


def bst(r):
    return datetime.datetime.utcfromtimestamp(int(r["createdOn"]) / 1000) + datetime.timedelta(hours=1)


rows.sort(key=lambda r: int(r["createdOn"]))
print(f"{len(rows)} SCORED submissions by us\n")
print(f"{'id':>9} {'created BST':>13}  {'DSC wt/tc/et/rc':<24} {'F1 wt/tc/et/rc':<24} name")
for r in rows:
    d = "/".join(g(r, f"lesionwise_dsc_mean_{x}") for x in ["wt", "tc", "et", "rc"])
    f1 = "/".join(g(r, f"small_instance_f1_{x}") for x in ["wt", "tc", "et", "rc"])
    print(f"{r['id']:>9} {bst(r).strftime('%m-%d %H:%M'):>13}  {d:<24} {f1:<24} {r['name'][:38]}")

print()
for day in ["2026-07-15", "2026-07-16", "2026-07-17"]:
    dd = datetime.date.fromisoformat(day)
    n = [r["id"] for r in rows if bst(r).date() == dd]
    print(f"  {day}: {len(n)} scored submission(s)  {n}")
print("\nNOTE: only SCORED rows appear. A submission made in the last ~hour may not be scored yet.")
