"""Search decode space against the OFFICIAL BraTS metric on local GT.

THE KEY INSIGHT (from reading metrics_parser.py, 2026-07-13):

    lesionwise_dsc: mean over LARGE GT instances of sq_dsc (0 if unmatched),
                    then  large_lesion_dsc.extend([0] * num_fp)   <-- EVERY FP APPENDS A ZERO
    small_f1      : 2*TP / (2*TP + num_fp + FN)                    <-- EVERY FP IN DENOMINATOR

  Both metrics are hurt by the SAME quantity: num_fp (region-wide FP components, ANY size,
  incl. a 2nd fragment of an already-matched lesion, since matching is 1:1 bipartite).
  => THERE IS NO INTRINSIC DSC<->F1 TRADEOFF. Both want: maximise TP, minimise FP.

  Observed on the leaderboard:
    9771379  mean(topk,comp)  hyst .30/.60 : DSC .718/.759/.740  F1 .366/.469/.476   (mr 54)
    9771380  mean(ccloss,topk) hyst .40/.70: DSC .670/.715/.691  F1 .429/.530/.543   (mr 131)
                                                                  ^^^ #3/1422 in the WHOLE comp
  CC-loss finds MORE TPs (F1 up) and MORE FPs (DSC down). The prize is a decode that keeps
  its TPs and kills its FPs -> both metrics rise together -> top-3 overall.

This sweeps that space and reports TP / FP / FN directly so the mechanism is visible.
"""
import os, sys, glob, json, tempfile
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
CFG = f"{HOME}/brats2025/brats_eval/brats_evaluation/configs/config_mets.yaml"
VOL_THRESH = 20.0
OVERLAP_THRESH = 0.1
REGIONS = ["wt", "tc", "et"]

# ---------------------------------------------------------------- decode space
# mode: plain | hyst | gate
#   gate: start from BASE decode, then ADD connected components from DONOR that the base
#         missed, but only if the donor component's peak prob >= gate_thresh. This is the
#         "keep CC-loss's extra detections, reject its junk" idea, made explicit.
V = []
# anchors with KNOWN leaderboard scores (validation gate)
V += [("REF hyst.30/.60 mean",  "mean_tc",  "hyst",  0.30, 0.60, None)]   # 9771379 team-best
V += [("REF hyst.40/.70 cclos", "mean_cct", "hyst",  0.40, 0.70, None)]   # 9771380 (#3 F1)
V += [("WIN 3way hyst.30/.70",  "mean3",    "hyst",  0.30, 0.70, None)]   # round-1 winner
# refine around the winner: is hi=0.70 the peak, and does lo move?
for lo in [0.20, 0.25, 0.30, 0.35]:
    for hi in [0.72, 0.75, 0.80, 0.85]:
        V.append((f"3way {lo:.2f}/{hi:.2f}", "mean3", "hyst", lo, hi, None))
# less ccloss dilution: weighted 3-way (topk+comp equal, ccloss down-weighted)
for w in [0.15, 0.20, 0.25]:
    for hi in [0.65, 0.70, 0.75]:
        V.append((f"w3 cc{w:.2f} .30/{hi:.2f}", f"w3_{w:.2f}", "hyst", 0.30, hi, None))
VARIANTS = V


def bases(cid):
    pt = np.load(f"{LE}/pred_topk/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{LE}/pred_completion/{cid}.npz")["probabilities"][:3]
    cc = np.load(f"{LE}/pred_ccloss/{cid}.npz")["probabilities"][:3]
    d = {"mean_tc": 0.5 * pt + 0.5 * cp,
         "mean_cct": 0.5 * cc + 0.5 * pt,
         "mean3": (pt + cp + cc) / 3.0}
    for w in (0.25, 0.35):
        d[f"w{w:.2f}_cct"] = (1 - w) * (0.5 * pt + 0.5 * cp) + w * cc
    for w in (0.15, 0.20, 0.25):
        d[f"w3_{w:.2f}"] = (1 - w) * (0.5 * pt + 0.5 * cp) + w * cc
    d["_ccloss"] = cc
    return d


def hyst_mask(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab]


def gate_mask(base_p, donor_p, lo, hi, gate):
    """base hysteresis, PLUS donor components (hyst at same lo/hi) whose peak >= gate
    and which do not already touch the base mask."""
    m = hyst_mask(base_p, lo, hi)
    dm = hyst_mask(donor_p, lo, hi)
    lab, n = cc_label(dm)
    if n == 0:
        return m
    for i in range(1, n + 1):
        comp = lab == i
        if m[comp].any():
            continue                      # already detected by base
        if donor_p[comp].max() >= gate:   # confident enough to trust
            m |= comp
    return m


def decode(B, base, mode, lo, hi, gate):
    avg = B[base]
    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):
        if mode == "plain":
            m = avg[ch] >= 0.5
        elif mode == "hyst":
            m = hyst_mask(avg[ch], lo, hi)
        else:
            m = gate_mask(avg[ch], B["_ccloss"][ch], lo, hi, gate)
        seg[m] = cls
    return seg


def parse_region(d):
    num_fp = d.get("fp", 0) or 0
    refs = d.get("reference_instances", []) or []
    large_dsc, large_nsd = [], []
    tp = fn = 0
    small_found = large_found = False
    for r in refs:
        vol = r.get("volume")
        if vol is None:
            continue
        is_large = vol >= VOL_THRESH
        large_found |= is_large
        small_found |= not is_large
        matched = r.get("is_matched") == 1
        sq = r.get("sq_dsc")
        if matched and is_large:
            large_dsc.append(sq); large_nsd.append(r.get("sq_nsd"))
        elif matched and not is_large:
            if sq is not None and sq >= OVERLAP_THRESH: tp += 1
            else: fn += 1
        elif (not matched) and is_large:
            large_dsc.append(0); large_nsd.append(0)
        else:
            fn += 1
    if num_fp > 0:
        large_dsc.extend([0] * num_fp); large_nsd.extend([0] * num_fp)
    o = {}
    if d.get("n_ref_instances", 0) == 0 or not large_found:
        o["dsc"] = o["nsd"] = np.nan
    else:
        o["dsc"] = float(np.mean(large_dsc)) if large_dsc else 0.0
        nz = [x for x in large_nsd if x is not None]
        o["nsd"] = float(np.mean(nz)) if nz else 0.0
    if small_found:
        den = 2 * tp + num_fp + fn
        o["f1"] = (2 * tp / den) if den > 0 else 0.0
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
        for name, base, mode, lo, hi, gate in VARIANTS:
            seg = decode(B, base, mode, lo, hi, gate)
            p = f"{td}/p.nii.gz"
            nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), p)
            try:
                g2r = ev.evaluate(p, gt)
                out[name] = {r: parse_region(g2r[r].to_dict(True)) for r in REGIONS if r in g2r}
            except Exception:
                out[name] = {r: {"dsc": np.nan, "nsd": np.nan, "f1": np.nan,
                                 "tp": np.nan, "fn": np.nan, "fp": np.nan} for r in REGIONS}
    return cid, out


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss/{c}.npz")]
    print(f"[score] {len(cases)} cases x {len(VARIANTS)} variants, nproc={NPROC}", flush=True)
    KEYS = ["dsc", "nsd", "f1", "tp", "fn", "fp"]
    acc = {v[0]: {r: {k: [] for k in KEYS} for r in REGIONS} for v in VARIANTS}
    with Pool(NPROC) as pool:
        for i, (cid, res) in enumerate(pool.imap_unordered(one_case, cases)):
            for vn, per in res.items():
                for r, m in per.items():
                    for k in KEYS:
                        acc[vn][r][k].append(m.get(k, np.nan))
            if (i + 1) % 10 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    def nm(xs):
        xs = [x for x in xs if x is not None]
        return float(np.nanmean(xs)) if len(xs) else float("nan")
    fmt = lambda xs: "/".join(f"{x:.3f}" for x in xs)

    rows = {}
    print(f"\n{'variant':<24} | {'DSC WT/TC/ET':^17} | {'NSD WT/TC/ET':^17} | {'F1 WT/TC/ET':^17} | {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 104)
    for name, *_ in VARIANTS:
        d = [nm(acc[name][r]["dsc"]) for r in REGIONS]
        s = [nm(acc[name][r]["nsd"]) for r in REGIONS]
        f = [nm(acc[name][r]["f1"]) for r in REGIONS]
        tp = nm(acc[name]["wt"]["tp"]); fp = nm(acc[name]["wt"]["fp"]); fn = nm(acc[name]["wt"]["fn"])
        rows[name] = {"dsc": d, "nsd": s, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        print(f"{name:<24} | {fmt(d)} | {fmt(s)} | {fmt(f)} | {tp:5.2f} {fp:5.2f} {fn:5.2f}")
    json.dump(rows, open(f"{LE}/decode_scores.json", "w"), indent=1)

    print("\n=== vs LOCAL anchors (REF hyst.30/.60 = 9771379 team-best) ===")
    ref = rows["REF hyst.30/.60 mean"]
    print(f"  ANCHOR REF hyst.30/.60 : DSC {fmt(ref['dsc'])}  NSD {fmt(ref['nsd'])}  F1 {fmt(ref['f1'])}  FP {ref['fp']:.2f}")
    dom = []
    for n, r in rows.items():
        if n.startswith("REF"): continue
        if all(r["dsc"][i] >= ref["dsc"][i] for i in range(3)) \
           and all(r["nsd"][i] >= ref["nsd"][i] for i in range(3)) \
           and all(r["f1"][i]  >= ref["f1"][i]  for i in range(3)):
            gain = sum(r["dsc"]) + sum(r["nsd"]) + sum(r["f1"]) - sum(ref["dsc"]) - sum(ref["nsd"]) - sum(ref["f1"])
            dom.append((gain, n, r))
    dom.sort(reverse=True)
    print(f"
  DECODES THAT DOMINATE THE TEAM-BEST ON ALL 9 TUMOUR METRICS: {len(dom)}")
    for gain, n, r in dom[:10]:
        print(f"   +{gain:.3f}  {n:<22} DSC {fmt(r['dsc'])}  NSD {fmt(r['nsd'])}  F1 {fmt(r['f1'])}  FP {r['fp']:.2f}")
    if not dom:
        print("   (none)")
    print("SCORE_COMPLETE")
