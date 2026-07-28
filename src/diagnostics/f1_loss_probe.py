"""Where do we LOSE small-lesion detections? Two suspects, measured directly.

SUSPECT 1 -- 50/50 PROBABILITY AVERAGING.
  A lesion found by one parent (p=0.9) and missed by the other (p=0.1) averages to 0.5
  and dies at threshold. Averaging is a recall killer for exactly the marginal,
  model-specific detections that small-instance F1 scores.
  Measure: n_cc(mean>0.5) vs n_cc(max>0.5), and how many max-CCs the mean loses entirely.

SUSPECT 2 -- hi=0.60 IN OUR HYSTERESIS DECODE.
  hyst keeps CCs of (p>lo) that contain a core (p>hi). With hi=0.60 we DISCARD any lesion
  whose peak probability lands in [0.50,0.60] -- i.e. we detect strictly LESS than a plain
  0.5 threshold would. Scored evidence: 9770654 (plain 0.5) F1 .370/.477/.484
                                     -> 9771150 (hi 0.60)  F1 .365/.468/.475.
  Measure: of the CCs a 0.5-threshold finds, how many have peak prob in [0.50,0.60]
  (= thrown away by hi=0.60), and how big are they.

Only components >= 27 vox are counted -- the metric's scoring threshold. Sub-27-vox
specks are invisible to F1, which is why hysteresis removing them changed nothing.
"""
import os, sys, glob, numpy as np
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

V = os.path.expanduser("~/brats2025/val_infer")
TOPK, COMP = f"{V}/pred_topk_s025", f"{V}/pred_topk_completion_s025"
NCASE = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 14
REG = ["WT", "TC", "ET"]
MINVOX = 27          # the metric's scoring threshold
cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))[:NCASE]


def ccs(mask, minvox=MINVOX):
    lab, n = cc_label(mask)
    if n == 0:
        return lab, np.array([], int)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.where(sizes[1:] >= minvox)[0] + 1
    return lab, keep


def one(cid):
    pt = np.load(f"{TOPK}/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{COMP}/{cid}.npz")["probabilities"][:3]
    mean = 0.5 * pt + 0.5 * cp
    mx = np.maximum(pt, cp)
    # per region: [n_mean, n_max, n_max_lost_by_mean, n_mean_peak_in_50_60, n_mean_total]
    out = np.zeros((3, 5), np.int64)
    for r in range(3):
        m, x = mean[r], mx[r]
        lab_m, keep_m = ccs(m > 0.5)
        lab_x, keep_x = ccs(x > 0.5)
        out[r, 0] = len(keep_m)
        out[r, 1] = len(keep_x)
        # max-CCs with NO overlap with any surviving mean-CC = detections averaging destroyed
        mean_fg = np.isin(lab_m, keep_m)
        lost = 0
        for i in keep_x:
            comp = lab_x == i
            if not mean_fg[comp].any():
                lost += 1
        out[r, 2] = lost
        # of the mean's own CCs, how many have peak prob in [0.50,0.60) -> killed by hi=0.60
        marg = 0
        for i in keep_m:
            comp = lab_m == i
            if m[comp].max() < 0.60:
                marg += 1
        out[r, 3] = marg
        out[r, 4] = len(keep_m)
    return out


if __name__ == "__main__":
    print(f"[f1probe] {len(cases)} cases, components >= {MINVOX} vox (the metric's threshold)\n", flush=True)
    A = np.zeros((3, 5), np.int64)
    with Pool(NPROC) as pool:
        for i, o in enumerate(pool.imap_unordered(one, cases, chunksize=1)):
            A += o
            if (i + 1) % 20 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    print("\n=== SUSPECT 1: 50/50 averaging vs max-fusion (detections lost to averaging) ===")
    for r in range(3):
        n_mean, n_max, lost = A[r, 0], A[r, 1], A[r, 2]
        print(f"  {REG[r]}:  mean>0.5 finds {n_mean:5d} instances |  max>0.5 finds {n_max:5d} "
              f"| averaging DESTROYS {lost:4d} instances that max would detect "
              f"({100*lost/max(n_max,1):.1f}% of max's)")

    print("\n=== SUSPECT 2: hi=0.60 discards marginal detections (peak prob in [0.50,0.60)) ===")
    for r in range(3):
        tot, marg = A[r, 4], A[r, 3]
        print(f"  {REG[r]}:  of {tot:5d} instances a 0.5-threshold finds, {marg:4d} have peak prob < 0.60 "
              f"-> THROWN AWAY by our hi=0.60 decode ({100*marg/max(tot,1):.1f}%)")
    print("\nPROBE_COMPLETE")
