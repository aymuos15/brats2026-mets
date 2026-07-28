import os, csv, synapseclient

tok = open(os.path.expanduser("~/brats2025/syn.tok")).read().strip()
syn = synapseclient.Synapse(silent=True)
syn.login(authToken=tok)
q = syn.tableQuery("SELECT * FROM syn74508245 WHERE submitterid = 3495063")
rows = list(csv.DictReader(open(q.filepath)))
rows.sort(key=lambda r: int(r["createdOn"]))


def g(r, m):
    s = (r.get(m) or "").strip()
    return f"{float(s):.3f}"[1:] if s else "  — "


print(f"{len(rows)} rows for us\n")
for r in rows[-4:]:
    d = "/".join(g(r, f"lesionwise_dsc_mean_{x}") for x in ["wt", "tc", "et", "rc"])
    f1 = "/".join(g(r, f"small_instance_f1_{x}") for x in ["wt", "tc", "et", "rc"])
    print(f"{r['id']}  {r['submission_status']:>10}  DSC {d}  F1 {f1}  | {r['name'][:40]}")
print("\n9772250 (slot1 today) in table:", any(r["id"] == "9772250" for r in rows))
for r in rows:
    if r["id"] == "9772250":
        print("  status:", r["submission_status"])
