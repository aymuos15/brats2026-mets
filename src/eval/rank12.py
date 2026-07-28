"""rank12 — mean-rank leaderboard over ALL 12 scored metrics.

Rebuilt 2026-07-17 (the original was lost; see handoff §0b).
Run:  ~/brats2025/.venv/bin/python rank12.py [lb_snapshot.csv]

12 METRICS, NOT 11 (handoff §0): 4 lesionwise DSC + 4 lesionwise NSD +
4 small_instance_f1, over wt/tc/et/rc. small_instance_f1_rc IS scored and we
LEAD it — dropping it is the old handoff's bug.

Convention: a submission is ranked on a metric only if it HAS a value there;
its mean rank averages the metrics it has. This is the parser's NaN-exclusion
and it is why a no-RC submission can lead (it is ranked on 9, not 12). The
zero-for-absent alternative is printed alongside, since it changes our place.
Ties share a rank (1 + strictly-better count).
"""
import sys, csv, os
from collections import defaultdict

OURS = "3495063"
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/brats2025/lb_snapshot.csv")

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
        subs.append({"id": r["id"], "team": r["submitterid"], "name": r.get("name", ""), "v": v})

print(f"[rank12] {len(subs)} scored submissions | {len(METRICS)} metrics")


def mean_ranks(absent_zero):
    """Rank each metric high-is-better; return {sub_id: mean rank}."""
    per = defaultdict(list)
    for m in METRICS:
        if absent_zero:
            vals = [(s["id"], s["v"].get(m, 0.0)) for s in subs]
        else:
            vals = [(s["id"], s["v"][m]) for s in subs if m in s["v"]]
        order = sorted(vals, key=lambda t: -t[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            rank = i + 1                      # ties share the better rank
            for k in range(i, j + 1):
                per[order[k][0]].append(rank)
            i = j + 1
    return {sid: sum(rs) / len(rs) for sid, rs in per.items()}, per


for absent_zero in (False, True):
    mr, per = mean_ranks(absent_zero)
    best = {}
    for s in subs:                            # a team takes its best submission
        t = s["team"]
        if t not in best or mr[s["id"]] < mr[best[t]["id"]]:
            best[t] = s
    board = sorted(best.values(), key=lambda s: mr[s["id"]])
    tag = "ZERO-FOR-ABSENT" if absent_zero else "EXCLUDE-ABSENT (official parser)"
    print(f"\n=== {tag} ===")
    print(f"{'#':>3} {'team':>9} {'mr':>7} {'best sub':>9}  {'nmet':>4}  DSC wt/tc/et/rc            F1 wt/tc/et/rc")
    for i, s in enumerate(board[:10], 1):
        f = lambda m: f"{s['v'][m]:.3f}"[1:] if m in s["v"] else "  — "
        d = "/".join(f(f"lesionwise_dsc_mean_{r}") for r in REG)
        f1 = "/".join(f(f"small_instance_f1_{r}") for r in REG)
        me = " <-- US" if s["team"] == OURS else ""
        print(f"{i:>3} {s['team']:>9} {mr[s['id']]:>7.1f} {s['id']:>9}  {len(s['v']):>4}  {d:<24}  {f1}{me}")
    for i, s in enumerate(board, 1):
        if s["team"] == OURS:
            print(f"    -> we are #{i} of {len(board)} teams | mr {mr[s['id']]:.1f} | best sub {s['id']}")
            if not absent_zero:
                print("    per-metric ranks (of scored subs having that metric):")
                for m, rk in zip(METRICS, per[s["id"]]):
                    print(f"      {m:<28} {s['v'][m]:.3f}  rank {rk}")
            break
print("\nRANK12_COMPLETE")
