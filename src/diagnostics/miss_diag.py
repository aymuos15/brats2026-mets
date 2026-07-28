"""Are the ~2.5 small lesions we miss per case RECOVERABLE, or is the model blind to them?

This decides what is worth doing next.

For every SMALL ground-truth lesion (<20 mm^3 -- the ones small-instance F1 scores), measure
the PEAK predicted probability inside it, in the 3-way ensemble we actually submit.
Then bucket:

  peak < 0.10   the network never saw it. No decode recovers this. Only better TRAINING.
  0.10 - 0.30   below even our LOW threshold. Essentially blind.
  0.30 - 0.70   PRESENT but below our seed (hi=0.70) -> RECOVERABLE BY DECODE.
  peak >= 0.70  seeded. We should already be detecting it (if we miss it, it is a
                matching/extent problem, not a detection one).

If most misses sit in the 0.30-0.70 band, the ceiling is our decode and a smarter/lower
seed wins. If most sit below 0.10, the ceiling is the model and only detection-oriented
TRAINING (targeted sampling, detection head) can move it.

Also reports, for contrast, the same distribution over FALSE-POSITIVE components -- because
if FPs and missed-lesions occupy the SAME probability band, then no threshold can separate
them and the decode really is finished.
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


def avg3(cid):
    pt = np.load(f"{LE}/pred_topk/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{LE}/pred_completion/{cid}.npz")["probabilities"][:3]
    cc = np.load(f"{LE}/pred_ccloss/{cid}.npz")["probabilities"][:3]
    return (pt + cp + cc) / 3.0


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
    P = avg3(cid)

    out = {r: {"miss_peaks": [], "hit_peaks": [], "fp_peaks": []} for r in REG}
    for r, (ch, labs) in REG.items():
        p = P[ch]
        pred = hyst_mask(p, LO, HI)          # what we actually submit
        gt = np.isin(a, labs)
        glab, gn = cc_label(gt, STRUCT)
        # --- every SMALL GT lesion: is it detected, and what is our peak inside it?
        for i in range(1, gn + 1):
            m = glab == i
            if m.sum() * vox_mm3 >= VOL_THRESH:
                continue                      # small instances only -- that is what F1 scores
            peak = float(p[m].max())
            detected = bool(pred[m].any())
            (out[r]["hit_peaks"] if detected else out[r]["miss_peaks"]).append(peak)
        # --- every predicted component with NO gt overlap: an FP. what is its peak?
        plab, pn = cc_label(pred, STRUCT)
        for i in range(1, pn + 1):
            m = plab == i
            if not gt[m].any():
                out[r]["fp_peaks"].append(float(p[m].max()))
    return out


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss/{c}.npz")]
    print(f"[miss] {len(cases)} cases | decode = 3-way hysteresis {LO}/{HI}", flush=True)
    agg = {r: {"miss_peaks": [], "hit_peaks": [], "fp_peaks": []} for r in REG}
    with Pool(NPROC) as pool:
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
        print(f"\n{'='*100}\n=== {r.upper()} — {tot} small GT lesions: "
              f"{len(hit)} detected ({100*len(hit)/max(tot,1):.0f}%), "
              f"{len(miss)} MISSED ({100*len(miss)/max(tot,1):.0f}%)")
        print(f"  peak prob inside MISSED lesions :\n   {hist(miss)}")
        print(f"  peak prob inside DETECTED ones  :\n   {hist(hit)}")
        print(f"  peak prob inside FALSE POSITIVES:\n   {hist(fp)}")
        rec = sum(1 for x in miss if 0.30 <= x < 0.70)
        blind = sum(1 for x in miss if x < 0.10)
        if miss:
            print(f"  --> {rec} of {len(miss)} misses ({100*rec/len(miss):.0f}%) sit in 0.30-0.70 "
                  f"= BELOW OUR SEED, but PRESENT  -> recoverable by decode")
            print(f"  --> {blind} of {len(miss)} misses ({100*blind/len(miss):.0f}%) peak below 0.10 "
                  f"= THE MODEL IS BLIND  -> only training helps")

    print(f"\n{'='*100}\n=== THE QUESTION: can a threshold separate missed lesions from false positives? ===")
    for r in REG:
        miss = [x for x in agg[r]["miss_peaks"] if x >= 0.30]
        fp = agg[r]["fp_peaks"]
        if not miss or not fp:
            continue
        print(f"  {r.upper()}: missed-lesion peaks (>=0.30) median {np.median(miss):.3f} | "
              f"FP peaks median {np.median(fp):.3f}")
        # if we dropped the seed to 0.50, what would we gain vs. let in?
        gain = sum(1 for x in agg[r]["miss_peaks"] if 0.50 <= x < 0.70)
        cost = sum(1 for x in fp if 0.50 <= x < 0.70)
        print(f"        seed 0.70 -> 0.50 would RECOVER {gain} lesions and ADMIT {cost} new FPs")
    print("MISS_DIAG_COMPLETE")
