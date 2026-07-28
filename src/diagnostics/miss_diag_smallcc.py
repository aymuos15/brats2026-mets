"""Did SmallCC shrink the 43% BLINDNESS?  (the ONLY question SmallCC exists to answer)

The handoff §1/§4: 43% of the small lesions we MISS have peak prob < 0.10 in the current
3-way ensemble -- the network never fired. SmallCC's whole point is to make the net FIRE on
those. Pseudo-Dice/EMA is BLIND to this (it scored CC-loss v1 at 0.917 while lesion-wise DSC
was 0.327). So we measure the blind bucket directly, comparing three probability sources:

  MODE=base3    mean(topk, completion, ccloss)                -- the CURRENT submission (baseline)
  MODE=smallcc  SmallCC alone                                 -- did the NETWORK stop being blind?
  MODE=fuse4    mean(topk, completion, ccloss, smallcc)       -- does it help the ENSEMBLE we ship?

`smallcc` is the diagnostic that isolates the training effect; `fuse4` is what we'd actually
submit. Run all three and compare the `peak < 0.10` line -- that number is the verdict.

Usage: python miss_diag_smallcc.py <base3|smallcc|fuse4> [nproc]
   or: python miss_diag_smallcc.py all [nproc]     # runs base3, smallcc, fuse4 in turn

NOTE: identical geometry, thresholds and small-lesion definition to miss_diag.py -- the ONLY
change is which .npz sources feed `probs()`. Decode is byte-identical (hysteresis 0.30/0.70),
so the blind-bucket numbers are directly comparable to the 43% already on record.
"""
import os, sys, json
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
VOL_THRESH = 20.0          # official small/large cut, in mm^3
LO, HI = 0.30, 0.70        # our submitted decode
STRUCT = np.ones((3, 3, 3), int)
REG = {"wt": (0, [1, 2, 3]), "tc": (1, [1, 3]), "et": (2, [3])}
BANDS = [(0.0, 0.10), (0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]

# which prediction dirs each mode averages (first 3 channels = wt/tc/et)
MODES = {
    "base3":   ["pred_topk", "pred_completion", "pred_ccloss"],
    "smallcc": ["pred_smallcc"],
    "fuse4":   ["pred_topk", "pred_completion", "pred_ccloss", "pred_smallcc"],
    "topk":    ["pred_topk"],        # SmallCC's BASE model, single -> controlled comparison vs smallcc
}
MODE = None  # set in __main__


def probs(cid):
    # nnU-Net stores .npz probabilities as (C, z, y, x) -- reversed spatial axes vs the
    # nibabel GT (x, y, z). Transpose spatial dims back so p aligns voxel-for-voxel with GT.
    acc = None
    for d in MODES[MODE]:
        p = np.load(f"{LE}/{d}/{cid}.npz")["probabilities"][:3]
        p = np.transpose(p, (0, 3, 2, 1))          # (C,z,y,x) -> (C,x,y,z)
        acc = p if acc is None else acc + p
    return acc / len(MODES[MODE])


def hyst_mask(p, lo, hi):
    lab, n = cc_label(p > lo, STRUCT)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab]


def one(cid):
    a = np.asarray(nib.load(f"{LE}/gt/{cid}.nii.gz").dataobj)
    img = nib.load(f"{LE}/gt/{cid}.nii.gz")
    vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    P = probs(cid)

    out = {r: {"miss_peaks": [], "hit_peaks": [], "fp_peaks": []} for r in REG}
    for r, (ch, labs) in REG.items():
        p = P[ch]
        pred = hyst_mask(p, LO, HI)
        gt = np.isin(a, labs)
        glab, gn = cc_label(gt, STRUCT)
        for i in range(1, gn + 1):
            m = glab == i
            if m.sum() * vox_mm3 >= VOL_THRESH:
                continue
            peak = float(p[m].max())
            detected = bool(pred[m].any())
            (out[r]["hit_peaks"] if detected else out[r]["miss_peaks"]).append(peak)
        plab, pn = cc_label(pred, STRUCT)
        for i in range(1, pn + 1):
            m = plab == i
            if not gt[m].any():
                out[r]["fp_peaks"].append(float(p[m].max()))
    return out


def run(nproc):
    ref = MODES[MODE][-1]  # every mode includes at least one dir; gate on the last one present
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if all(os.path.exists(f"{LE}/{d}/{c}.npz") for d in MODES[MODE])]
    print(f"\n########## MODE={MODE}  ({' + '.join(MODES[MODE])}) ##########", flush=True)
    print(f"[miss] {len(cases)} cases | decode = hysteresis {LO}/{HI}", flush=True)
    if not cases:
        print("  (no cases -- predictions missing for this mode; run predict_smallcc.sh first)")
        return
    agg = {r: {"miss_peaks": [], "hit_peaks": [], "fp_peaks": []} for r in REG}
    with Pool(nproc) as pool:
        for i, o in enumerate(pool.imap_unordered(one, cases)):
            for r in REG:
                for k in agg[r]:
                    agg[r][k] += o[r][k]
            if (i + 1) % 20 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    def hist(xs):
        n = len(xs)
        if n == 0:
            return "  (none)"
        return "  ".join(f"{lo:.2f}-{hi:.2f}: {sum(1 for x in xs if lo <= x < hi):4d} "
                         f"({100*sum(1 for x in xs if lo <= x < hi)/n:4.1f}%)"
                         for lo, hi in BANDS)

    for r in REG:
        miss, hit, fp = agg[r]["miss_peaks"], agg[r]["hit_peaks"], agg[r]["fp_peaks"]
        tot = len(miss) + len(hit)
        print(f"\n{'='*100}\n=== [{MODE}] {r.upper()} — {tot} small GT lesions: "
              f"{len(hit)} detected ({100*len(hit)/max(tot,1):.0f}%), "
              f"{len(miss)} MISSED ({100*len(miss)/max(tot,1):.0f}%)")
        print(f"  peak prob inside MISSED lesions :\n   {hist(miss)}")
        print(f"  peak prob inside DETECTED ones  :\n   {hist(hit)}")
        print(f"  peak prob inside FALSE POSITIVES:\n   {hist(fp)}")
        blind = sum(1 for x in miss if x < 0.10)
        rec = sum(1 for x in miss if 0.30 <= x < 0.70)
        if miss:
            print(f"  --> BLIND (peak<0.10): {blind}/{len(miss)} ({100*blind/len(miss):.0f}%)  "
                  f"[base3 record = 43%]   |   recoverable 0.30-0.70: {rec}/{len(miss)} "
                  f"({100*rec/len(miss):.0f}%)")
    print(f"MISS_DIAG_COMPLETE mode={MODE}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in list(MODES) + ["all"]:
        sys.exit(f"usage: python {sys.argv[0]} <base3|smallcc|fuse4|topk|all> [nproc]")
    which = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    for m in (["base3", "smallcc", "fuse4"] if which == "all" else [which]):
        MODE = m
        run(nproc)
