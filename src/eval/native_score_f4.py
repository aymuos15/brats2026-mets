"""FAST native fuse4 vs base3 scorer — replicates the official small-F1 + lesionwise-DSC
logic (from score_decodes.parse_region) natively, NO panoptica/NSD (which hangs on big
surfaces). Runs at low nproc so it does not starve D015's dataloaders on the same box.

Official structure reproduced:
  small_f1 = 2*TP / (2*TP + num_fp + FN)          # small GT instances; num_fp = ALL region FP comps
  lesion_dsc = mean over LARGE GT instances of matched DSC (0 if unmatched), + one 0 per FP
  match: 1:1 greedy by max overlap; small counts TP iff matched DSC >= 0.1
"""
import os, sys, json
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
VOL_THRESH = 20.0
OVL = 0.1
STRUCT = np.ones((3, 3, 3), int)
REG = {"wt": (0, [1, 2, 3]), "tc": (1, [1, 3]), "et": (2, [3])}

def load3(cid, d):
    return np.transpose(np.load(f"{LE}/{d}/{cid}.npz")["probabilities"][:3], (0, 3, 2, 1))

# fusion candidates
def cands(cid):
    pt = load3(cid, "pred_topk"); cp = load3(cid, "pred_completion")
    cc = load3(cid, "pred_ccloss"); sc = load3(cid, "pred_smallcc")
    b3 = (pt + cp + cc) / 3.0
    return {"base3": b3,
            "fuse4eq": (pt + cp + cc + sc) / 4.0,
            "fuse4sc5": (pt + cp + cc + 0.5 * sc) / 3.5}

VARIANTS = [("base3 .30/.70", "base3", 0.30, 0.70)]
for hi in (0.65, 0.70, 0.75):
    VARIANTS.append((f"fuse4eq .30/{hi:.2f}", "fuse4eq", 0.30, hi))
    VARIANTS.append((f"fuse4sc5 .30/{hi:.2f}", "fuse4sc5", 0.30, hi))

def hyst(p, lo, hi):
    lab, n = cc_label(p > lo, STRUCT)
    if n == 0:
        return np.zeros(p.shape, bool)
    cm = np.zeros(n + 1, np.float32); np.maximum.at(cm, lab.ravel(), p.ravel())
    keep = cm > hi; keep[0] = False
    return keep[lab]

def score_region(pred, gt, vox):
    glab, gn = cc_label(gt, STRUCT)
    plab, pn = cc_label(pred, STRUCT)
    gsz = np.bincount(glab.ravel()); psz = np.bincount(plab.ravel())
    used = set(); large_dsc = []; tp = fn = 0
    # match GT instances (largest first) to best-overlapping pred comp
    order = sorted(range(1, gn + 1), key=lambda i: -gsz[i])
    for i in order:
        gm = glab == i
        ov = plab[gm]; ov = ov[ov > 0]
        best = 0; bj = 0
        if ov.size:
            js, cts = np.unique(ov, return_counts=True)
            for j, c in zip(js, cts):
                if j in used: continue
                if c > best: best, bj = c, j
        is_large = gsz[i] * vox >= VOL_THRESH
        if bj:
            used.add(bj)
            dsc = 2 * best / (gsz[i] + psz[bj])
            if is_large: large_dsc.append(dsc)
            else: (tp := tp + 1) if dsc >= OVL else (fn := fn + 1)
        else:
            if is_large: large_dsc.append(0.0)
            else: fn += 1
    num_fp = pn - len(used)
    large_dsc.extend([0.0] * num_fp)
    dsc = float(np.mean(large_dsc)) if large_dsc else np.nan
    den = 2 * tp + num_fp + fn
    f1 = (2 * tp / den) if den > 0 else np.nan
    has_small = (tp + fn) > 0 or any(gsz[i] * vox < VOL_THRESH for i in range(1, gn + 1))
    return dsc, (f1 if has_small else np.nan), tp, num_fp, fn

def one(cid):
    img = nib.load(f"{LE}/gt/{cid}.nii.gz"); a = np.asarray(img.dataobj)
    vox = float(np.prod(img.header.get_zooms()[:3]))
    C = cands(cid)
    out = {}
    for name, base, lo, hi in VARIANTS:
        P = C[base]; per = {}
        for r, (ch, labs) in REG.items():
            per[r] = score_region(hyst(P[ch], lo, hi), np.isin(a, labs), vox)
        out[name] = per
    return out

if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_smallcc/{c}.npz")]
    print(f"[native] {len(cases)} cases x {len(VARIANTS)} variants, nproc={NPROC}", flush=True)
    agg = {v[0]: {r: {"dsc": [], "f1": [], "tp": [], "fp": [], "fn": []} for r in REG} for v in VARIANTS}
    with Pool(NPROC) as pool:
        for k, out in enumerate(pool.imap_unordered(one, cases)):
            for name in agg:
                for r in REG:
                    dsc, f1, tp, fp, fn = out[name][r]
                    agg[name][r]["dsc"].append(dsc); agg[name][r]["f1"].append(f1)
                    agg[name][r]["tp"].append(tp); agg[name][r]["fp"].append(fp); agg[name][r]["fn"].append(fn)
            if (k + 1) % 20 == 0:
                print(f"  ..{k+1}/{len(cases)}", flush=True)
    nm = lambda xs: float(np.nanmean(xs)) if len(xs) else float("nan")
    f3 = lambda xs: "/".join(f"{x:.3f}" for x in xs)
    ref = None
    print(f"\n{'variant':<20} | {'DSC wt/tc/et':^17} | {'F1 wt/tc/et':^17} | {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-"*82)
    rows = {}
    for name, *_ in VARIANTS:
        d = [nm(agg[name][r]["dsc"]) for r in REG]; f = [nm(agg[name][r]["f1"]) for r in REG]
        tp = nm(agg[name]["wt"]["tp"]); fp = nm(agg[name]["wt"]["fp"]); fn = nm(agg[name]["wt"]["fn"])
        rows[name] = (d, f);
        print(f"{name:<20} | {f3(d)} | {f3(f)} | {tp:5.2f} {fp:5.2f} {fn:5.2f}")
    d0, f0 = rows["base3 .30/.70"]
    print(f"\n=== vs base3 (=current best 9771508) — dominates on all 6 tumour DSC+F1? ===")
    for name, *_ in VARIANTS:
        if name == "base3 .30/.70": continue
        d, f = rows[name]
        dom = all(d[i] >= d0[i]-1e-9 for i in range(3)) and all(f[i] >= f0[i]-1e-9 for i in range(3))
        gain = sum(d)-sum(d0)+sum(f)-sum(f0)
        print(f"  {name:<20} {'DOMINATES' if dom else 'no'}  net{gain:+.3f}  "
              f"dDSC {f3([d[i]-d0[i] for i in range(3)])}  dF1 {f3([f[i]-f0[i] for i in range(3)])}")
    print("NATIVE_SCORE_COMPLETE")
