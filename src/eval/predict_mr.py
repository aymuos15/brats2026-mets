"""Predict the LEADERBOARD MEAN RANK of each candidate decode.

"Dominates on all 9 metrics" is a sufficient condition for improvement, not the objective.
The objective is mean rank over 11 metrics -- 8 DSC/NSD vs only 3 F1 -- so a decode that
trades a little F1 for more DSC could rank better despite not dominating.

Method:
  1. Four decodes have BOTH a local score (harness) and an official score (leaderboard):
        REF plain.5 mean      <-> 9770654
        REF hyst.40/.60 mean  <-> 9771150
        REF hyst.30/.60 mean  <-> 9771379   (team best)
        REF hyst.40/.70 cclos <-> 9771380   (#3 F1)
     The local set is small-lesion-enriched, so local values are offset from official ones --
     but monotonically. Fit leaderboard = a*local + b per metric on these 4 anchors.
  2. Map every candidate decode's local scores through that fit -> predicted official scores.
  3. RC is identical across all our decodes (CC-loss donor, DSC .630 / NSD .521) -- carry it.
  4. Rank each predicted submission against the 1,529 real scored submissions on all 11
     metrics, average the ranks -> predicted mean rank. Same method as the live standing.
"""
import json
import numpy as np

SP = "/home/ssk23/brats2025"
R1 = json.load(open(SP + "/decode_scores.json"))    # round-1 sweep (has plain.5 + hyst.40/.60)
R2 = json.load(open(SP + "/decode_scores2.json"))   # round-2 sweep (the grid)
LB = json.load(open(SP + "/lb.json"))

# metric order used everywhere: [wt, tc, et]
ANCHORS = [
    # local-key, which-sweep, official DSC wt/tc/et, NSD wt/tc/et, F1 wt/tc/et
    ("REF plain.5 mean",     R1, [.678, .718, .696], [.704, .775, .766], [.370, .477, .484]),
    ("REF hyst.40/.60 mean", R1, [.711, .753, .729], [.734, .807, .795], [.365, .468, .475]),
    ("REF hyst.30/.60 mean", R1, [.718, .759, .740], [.738, .811, .803], [.366, .469, .476]),
    ("REF hyst.40/.70 cclos", R1, [.670, .715, .691], [.692, .764, .752], [.429, .530, .543]),
]
RC_DSC, RC_NSD = 0.630, 0.521      # identical in every decode we would submit

# ---- 1. fit local -> official, per metric ---------------------------------
fits = {}
for fam in ("dsc", "nsd", "f1"):
    for i, reg in enumerate(["wt", "tc", "et"]):
        xs, ys = [], []
        for key, src, d, n, f in ANCHORS:
            off = {"dsc": d, "nsd": n, "f1": f}[fam][i]
            xs.append(src[key][fam][i])
            ys.append(off)
        a, b = np.polyfit(xs, ys, 1)
        pred = [a * x + b for x in xs]
        resid = max(abs(p - y) for p, y in zip(pred, ys))
        fits[(fam, reg)] = (a, b, resid)

print("=== local -> leaderboard fit (4 anchors) ===")
for k, (a, b, r) in fits.items():
    print(f"  {k[0]}_{k[1]:<3} official = {a:6.3f} * local + {b:+.3f}   max residual {r:.4f}")
worst = max(r for _, _, r in fits.values())
print(f"  worst residual across all 9 metrics: {worst:.4f}  "
      f"({'GOOD - fit is tight' if worst < 0.012 else 'LOOSE - treat predictions as rough'})")

# ---- 2. the real leaderboard distribution --------------------------------
M = ["lesionwise_dsc_mean_wt", "lesionwise_dsc_mean_tc", "lesionwise_dsc_mean_et", "lesionwise_dsc_mean_rc",
     "lesionwise_nsd_mean_wt", "lesionwise_nsd_mean_tc", "lesionwise_nsd_mean_et", "lesionwise_nsd_mean_rc",
     "small_instance_f1_wt", "small_instance_f1_tc", "small_instance_f1_et"]


def fn(v):
    try:
        return float(v)
    except Exception:
        return None


scored = [r for r in LB if r.get("submission_status") == "SCORED"]
dist = {m: sorted([v for v in (fn(r.get(m)) for r in scored) if v is not None], reverse=True)
        for m in M}
print(f"\nranking against {len(scored)} scored submissions")


def mean_rank(pred):
    """pred: dict metric -> value. Rank = 1 + #submissions strictly better."""
    ranks = []
    for m in M:
        v = pred[m]
        col = dist[m]
        ranks.append(1 + sum(1 for x in col if x > v))
    return float(np.mean(ranks)), ranks


def predict(local):
    p = {}
    for i, reg in enumerate(["wt", "tc", "et"]):
        a, b, _ = fits[("dsc", reg)]
        p[f"lesionwise_dsc_mean_{reg}"] = a * local["dsc"][i] + b
        a, b, _ = fits[("nsd", reg)]
        p[f"lesionwise_nsd_mean_{reg}"] = a * local["nsd"][i] + b
        a, b, _ = fits[("f1", reg)]
        p[f"small_instance_f1_{reg}"] = a * local["f1"][i] + b
    p["lesionwise_dsc_mean_rc"] = RC_DSC
    p["lesionwise_nsd_mean_rc"] = RC_NSD
    return p


# ---- 3. sanity: does the pipeline reproduce the KNOWN mean ranks? ---------
print("\n=== SANITY: predicted vs ACTUAL mean rank of the 4 anchors ===")
ACTUAL_MR = {"REF plain.5 mean": 158.2, "REF hyst.40/.60 mean": 75.4,
             "REF hyst.30/.60 mean": 55.7, "REF hyst.40/.70 cclos": 135.1}
ok = True
for key, src, *_ in ANCHORS:
    mr, _ = mean_rank(predict(src[key]))
    act = ACTUAL_MR[key]
    err = mr - act
    ok &= abs(err) < 20
    print(f"  {key:<22} predicted {mr:6.1f}   actual {act:6.1f}   err {err:+6.1f}")
print("  => pipeline reproduces known mean ranks" if ok else
      "  => WARNING: predictions are off; treat the ordering below as indicative only")

# ---- 4. rank every candidate ---------------------------------------------
cands = {}
for name, v in R2.items():
    if name.startswith("REF"):
        continue
    cands[name] = v
for name, v in R1.items():
    if name.startswith("REF") or name in cands:
        continue
    cands[name] = v

rows = []
for name, v in cands.items():
    if any(np.isnan(x) for x in v["dsc"] + v["nsd"] + v["f1"]):
        continue
    mr, _ = mean_rank(predict(v))
    rows.append((mr, name, v))
rows.sort()

base_mr = 55.7
print(f"\n=== PREDICTED LEADERBOARD MEAN RANK  (team-best 9771379 = {base_mr}) ===")
print(f"{'predicted mr':>12}  {'vs best':>8}  {'decode':<24} DSC wt/tc/et        F1 wt/tc/et")
print("-" * 96)
for mr, name, v in rows[:14]:
    d = "/".join(f"{x:.3f}" for x in v["dsc"])
    f = "/".join(f"{x:.3f}" for x in v["f1"])
    flag = "  <<<< BEST" if mr == rows[0][0] else ""
    print(f"{mr:12.1f}  {mr-base_mr:+8.1f}  {name:<24} {d}   {f}{flag}")

print("\n=== what rank would that be? ===")
top = rows[0]
teams = {}
mrmap = {}
for r in scored:
    vals = {m: fn(r.get(m)) for m in M}
    if all(v is None for v in vals.values()):
        continue
    rk = []
    for m in M:
        if vals[m] is None:
            continue
        rk.append(1 + sum(1 for x in dist[m] if x > vals[m]))
    if rk:
        mrmap[r["id"]] = np.mean(rk)
        t = r["submitterid"]
        if t not in teams or mrmap[r["id"]] < teams[t]:
            teams[t] = mrmap[r["id"]]
order = sorted(teams.values())
new = top[0]
pos = 1 + sum(1 for v in order if v < new and v != teams.get("3495063"))
print(f"  best candidate '{top[1]}' -> predicted mr {new:.1f}")
print(f"  that would place us ~#{pos} / {len(order)} teams (currently #7)")
