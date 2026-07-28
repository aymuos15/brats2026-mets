"""Did the CC-loss v2 loss fix work?

v1 (blob loss, no precision term) + hysteresis = 9771380 = F1 rank 3 / 1422, but sunk
by false positives. v2 adds a per-component FP penalty at the SAME weight as the recall
term (v1's boundary term was ~2400x too weak to bite).

The question is NOT "is v2's DSC higher" -- it is:
      did FP/case go DOWN while TP/case stayed UP?
Because the official scorer charges every FP to BOTH lesionwise DSC and small-instance F1,
that is the only combination that wins. If TP fell with FP, the penalty ate real detections
and lambda_fp is too high.

Compared head-to-head on the same 80 GT cases, same decode:
  ccloss v1  standalone   |  ccloss v2  standalone
  3-way with v1 (= the staged slot-1 submission)  |  3-way with v2  |  4-way with both
"""
import os, sys, json, tempfile
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
CFG = f"{HOME}/brats2025/brats_eval/brats_evaluation/configs/config_mets.yaml"
VOL_THRESH, OVERLAP_THRESH = 20.0, 0.1
REGIONS = ["wt", "tc", "et"]

VARIANTS = [
    ("REF 3way v1  .30/.70", "mean3_v1", 0.30, 0.70),   # the staged slot-1 submission
    ("REF hyst.30/.60 mean", "mean_tc",  0.30, 0.60),   # 9771379 team best (anchor)
    ("v1 alone    .40/.70", "cct_v1",   0.40, 0.70),    # 9771380 base (#3 F1)
    ("v2 alone    .40/.70", "cct_v2",   0.40, 0.70),    # <- direct A/B vs the line above
    ("v2 alone    .30/.60", "cct_v2",   0.30, 0.60),
    ("v2 alone    .30/.70", "cct_v2",   0.30, 0.70),
    ("3way v2     .30/.70", "mean3_v2", 0.30, 0.70),    # <- v2 swapped into the winner
    ("3way v2     .30/.60", "mean3_v2", 0.30, 0.60),
    ("3way v2     .30/.75", "mean3_v2", 0.30, 0.75),
    ("4way v1+v2  .30/.70", "mean4",    0.30, 0.70),    # both blob models in
    ("4way v1+v2  .30/.75", "mean4",    0.30, 0.75),
]


def bases(cid):
    pt = np.load(f"{LE}/pred_topk/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{LE}/pred_completion/{cid}.npz")["probabilities"][:3]
    c1 = np.load(f"{LE}/pred_ccloss/{cid}.npz")["probabilities"][:3]
    c2 = np.load(f"{LE}/pred_ccloss_v2/{cid}.npz")["probabilities"][:3]
    return {
        "mean_tc":  0.5 * pt + 0.5 * cp,
        "cct_v1":   0.5 * c1 + 0.5 * pt,
        "cct_v2":   0.5 * c2 + 0.5 * pt,
        "mean3_v1": (pt + cp + c1) / 3.0,
        "mean3_v2": (pt + cp + c2) / 3.0,
        "mean4":    (pt + cp + c1 + c2) / 4.0,
    }


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab]


def parse_region(d):
    num_fp = d.get("fp", 0) or 0
    refs = d.get("reference_instances", []) or []
    ld, ln = [], []
    tp = fn = 0
    small = large = False
    for r in refs:
        v = r.get("volume")
        if v is None:
            continue
        is_l = v >= VOL_THRESH
        large |= is_l
        small |= not is_l
        mt = r.get("is_matched") == 1
        sq = r.get("sq_dsc")
        if mt and is_l:
            ld.append(sq); ln.append(r.get("sq_nsd"))
        elif mt and not is_l:
            tp += 1 if (sq is not None and sq >= OVERLAP_THRESH) else 0
            fn += 0 if (sq is not None and sq >= OVERLAP_THRESH) else 1
        elif (not mt) and is_l:
            ld.append(0); ln.append(0)
        else:
            fn += 1
    if num_fp:
        ld.extend([0] * num_fp); ln.extend([0] * num_fp)
    o = {}
    if d.get("n_ref_instances", 0) == 0 or not large:
        o["dsc"] = o["nsd"] = np.nan
    else:
        o["dsc"] = float(np.mean(ld)) if ld else 0.0
        nz = [x for x in ln if x is not None]
        o["nsd"] = float(np.mean(nz)) if nz else 0.0
    if small:
        den = 2 * tp + num_fp + fn
        o["f1"] = (2 * tp / den) if den else 0.0
        o["tp"], o["fn"], o["fp"] = tp, fn, num_fp
    else:
        o["f1"] = o["tp"] = o["fn"] = o["fp"] = np.nan
    return o


_EV = None
def _ev():
    global _EV
    if _EV is None:
        from panoptica import Panoptica_Evaluator
        _EV = Panoptica_Evaluator.load_from_config(CFG)
    return _EV


def one_case(cid):
    import warnings; warnings.filterwarnings("ignore")
    ev = _ev(); B = bases(cid)
    gt = f"{LE}/gt/{cid}.nii.gz"
    ref = nib.load(gt)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, base, lo, hi in VARIANTS:
            avg = B[base]
            seg = np.zeros(avg.shape[1:], np.uint8)
            for ch, cls in zip([0, 1, 2], [2, 1, 3]):
                seg[hyst(avg[ch], lo, hi)] = cls
            p = f"{td}/p.nii.gz"
            nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), p)
            try:
                g2r = ev.evaluate(p, gt)
                out[name] = {r: parse_region(g2r[r].to_dict(True)) for r in REGIONS if r in g2r}
            except Exception:
                out[name] = {r: {k: np.nan for k in ("dsc", "nsd", "f1", "tp", "fn", "fp")}
                             for r in REGIONS}
    return cid, out


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss_v2/{c}.npz")]
    print(f"[v2] {len(cases)} cases x {len(VARIANTS)} variants", flush=True)
    KEYS = ["dsc", "nsd", "f1", "tp", "fn", "fp"]
    acc = {v[0]: {r: {k: [] for k in KEYS} for r in REGIONS} for v in VARIANTS}
    with Pool(NPROC) as pool:
        for i, (cid, res) in enumerate(pool.imap_unordered(one_case, cases)):
            for vn, per in res.items():
                for r, m in per.items():
                    for k in KEYS:
                        acc[vn][r][k].append(m.get(k, np.nan))
            if (i + 1) % 20 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    def nm(xs):
        xs = [x for x in xs if x is not None]
        return float(np.nanmean(xs)) if len(xs) else float("nan")
    fmt = lambda xs: "/".join(f"{x:.3f}" for x in xs)

    rows = {}
    hdr = ("variant".ljust(22) + " | " + "DSC WT/TC/ET".center(17) + " | " +
           "F1 WT/TC/ET".center(17) + " |    TP    FP    FN")
    print("\n" + hdr); print("-" * len(hdr))
    for name, *_ in VARIANTS:
        d = [nm(acc[name][r]["dsc"]) for r in REGIONS]
        f = [nm(acc[name][r]["f1"]) for r in REGIONS]
        s = [nm(acc[name][r]["nsd"]) for r in REGIONS]
        tp, fp, fn = (nm(acc[name]["wt"][k]) for k in ("tp", "fp", "fn"))
        rows[name] = {"dsc": d, "nsd": s, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        print(f"{name:<22} | {fmt(d)} | {fmt(f)} | {tp:5.2f} {fp:5.2f} {fn:5.2f}")
    json.dump(rows, open(f"{LE}/v2_scores.json", "w"), indent=1)

    print("\n=== THE VERDICT: v1 vs v2, same decode (.40/.70), same cases ===")
    a, b = rows["v1 alone    .40/.70"], rows["v2 alone    .40/.70"]
    print(f"  FP/case   v1 {a['fp']:.2f}  ->  v2 {b['fp']:.2f}   ({b['fp']-a['fp']:+.2f})")
    print(f"  TP/case   v1 {a['tp']:.2f}  ->  v2 {b['tp']:.2f}   ({b['tp']-a['tp']:+.2f})")
    print(f"  FN/case   v1 {a['fn']:.2f}  ->  v2 {b['fn']:.2f}   ({b['fn']-a['fn']:+.2f})")
    print(f"  DSC  wt   v1 {a['dsc'][0]:.3f} ->  v2 {b['dsc'][0]:.3f}  ({b['dsc'][0]-a['dsc'][0]:+.3f})")
    print(f"  F1   wt   v1 {a['f1'][0]:.3f} ->  v2 {b['f1'][0]:.3f}  ({b['f1'][0]-a['f1'][0]:+.3f})")
    if b["fp"] < a["fp"] - 0.05 and b["tp"] > a["tp"] - 0.05:
        print("  => THE FIX WORKED: fewer false positives, detections held.")
    elif b["fp"] < a["fp"] and b["tp"] < a["tp"] - 0.1:
        print("  => lambda_fp TOO HIGH: it bought precision by eating real detections.")
    else:
        print("  => the penalty did not bite. Likely because it targets the component MEAN,")
        print("     while hysteresis only ever reads the component PEAK.")

    print("\n=== vs the staged slot-1 submission (3-way with v1) ===")
    ref3 = rows["REF 3way v1  .30/.70"]
    print(f"  anchor 3way-v1 : DSC {fmt(ref3['dsc'])}  F1 {fmt(ref3['f1'])}  FP {ref3['fp']:.2f}")
    for n, r in sorted(rows.items(), key=lambda x: -(sum(x[1]["dsc"]) + sum(x[1]["f1"]))):
        if n.startswith("REF"):
            continue
        dom = (all(r["dsc"][i] >= ref3["dsc"][i] for i in range(3)) and
               all(r["f1"][i] >= ref3["f1"][i] for i in range(3)))
        print(f"  {n:<22} DSC {fmt(r['dsc'])}  F1 {fmt(r['f1'])}  FP {r['fp']:.2f}"
              + ("   <<< BEATS SLOT-1 ON ALL SIX" if dom else ""))
    print("SCORE_V2_COMPLETE")
