"""Mean rank of specific submission ids within the current snapshot."""
import sys, csv, os
from collections import defaultdict

TARGETS = sys.argv[1:] or ["9771992", "9772250", "9771508"]
csv_path = os.path.expanduser("~/brats2025/lb_snapshot.csv")
REG = ["wt", "tc", "et", "rc"]
METRICS = ([f"lesionwise_dsc_mean_{r}" for r in REG]
           + [f"lesionwise_nsd_mean_{r}" for r in REG]
           + [f"small_instance_f1_{r}" for r in REG])

rows = list(csv.DictReader(open(csv_path)))
subs = []
for r in rows:
    v = {}
    for m in METRICS:
        s = (r.get(m) or "").strip()
        if s:
            try:
                v[m] = float(s)
            except ValueError:
                pass
    if v:
        subs.append({"id": r["id"], "v": v})

per = defaultdict(list)
for m in METRICS:
    vals = [(s["id"], s["v"][m]) for s in subs if m in s["v"]]
    order = sorted(vals, key=lambda t: -t[1])
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        for k in range(i, j + 1):
            per[order[k][0]].append(i + 1)
        i = j + 1

mr = {sid: sum(rs) / len(rs) for sid, rs in per.items()}
print(f"snapshot: {len(subs)} scored subs with metrics\n")
for t in TARGETS:
    if t in mr:
        print(f"  {t}: mean rank {mr[t]:.2f}   per-metric {per[t]}")
    else:
        print(f"  {t}: not scored yet")
print("\n(metric order: DSC wt/tc/et/rc, NSD wt/tc/et/rc, F1 wt/tc/et/rc)")
