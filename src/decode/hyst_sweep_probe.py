"""Grid-probe hysteresis lo/hi over TWO tumour bases (parallel over cases).

base A "topk_comp"   = 0.5*TopK4reg + 0.5*completion  -> base of best sub 9771150
                        (DSC .711/.753/.729   F1 .365/.468/.475)
base B "ccloss_topk" = 0.5*CCloss   + 0.5*TopK4reg    -> base of sub 9770752
                        (DSC .600/.650/.628   F1 .400/.499/.500  <- top-6 F1 band)

Leaderboard read (07-13): our DSC/NSD already ties #5 (mr 25.6); the ENTIRE mr gap
is small-instance F1. So the decisive column is n_cc / n_small (instances kept),
NOT vox. Lowering `lo` grows CCs and can MERGE two true lesions into one -> that
destroys an instance -> costs F1. Watch n_cc.

No GT for the val set (labels withheld) -> characterises the surface, cannot score it.
"""
import os, sys, glob, numpy as np
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

V = os.path.expanduser("~/brats2025/val_infer")
TOPK, COMP, CCL = f"{V}/pred_topk_s025", f"{V}/pred_topk_completion_s025", f"{V}/pred_ccloss_s025"
NCASE = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 12

LOS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
HIS = [0.50, 0.60, 0.70]
REG = ["WT", "TC", "ET"]
BASES = ["topk_comp", "ccloss_topk"]
SMALL = 100
KEYS = [(b, lo, hi) for b in BASES for lo in LOS for hi in HIS]
KIDX = {k: i for i, k in enumerate(KEYS)}

cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(TOPK + "/*.npz"))[:NCASE]


def one(cid):
    """-> (acc[len(KEYS),3,4], ref[2,3,3])  for a single case."""
    a = np.zeros((len(KEYS), 3, 4), np.int64)
    rf = np.zeros((2, 3, 3), np.int64)
    pt = np.load(f"{TOPK}/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{COMP}/{cid}.npz")["probabilities"][:3]
    cc = np.load(f"{CCL}/{cid}.npz")["probabilities"][:3]
    avgs = {"topk_comp": 0.5 * pt + 0.5 * cp, "ccloss_topk": 0.5 * cc + 0.5 * pt}
    for bi, b in enumerate(BASES):
        for r in range(3):
            p = avgs[b][r]
            lab5, n5 = cc_label(p > 0.5)
            if n5:
                s5 = np.bincount(lab5.ravel())[1:]
                rf[bi, r] += [n5, int((s5 < SMALL).sum()), int(s5.sum())]
            for lo in LOS:
                lab, n = cc_label(p > lo)
                if n == 0:
                    continue
                sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                cmax = np.zeros(n + 1, np.float32)
                np.maximum.at(cmax, lab.ravel(), p.ravel())
                cmax = cmax[1:]
                for hi in HIS:
                    keep = cmax > hi
                    a[KIDX[(b, lo, hi)], r] += [int(keep.sum()),
                                                int((keep & (sizes < SMALL)).sum()),
                                                int(sizes[keep].sum()),
                                                int((~keep).sum())]
    return a, rf


if __name__ == "__main__":
    print(f"[probe] {len(cases)} cases | lo={LOS} hi={HIS} | bases={BASES} | nproc={NPROC}", flush=True)
    A = np.zeros((len(KEYS), 3, 4), np.int64)
    R = np.zeros((2, 3, 3), np.int64)
    with Pool(NPROC) as pool:
        for i, (a, rf) in enumerate(pool.imap_unordered(one, cases, chunksize=1)):
            A += a
            R += rf
            if (i + 1) % 10 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    b0 = A[KIDX[("topk_comp", 0.40, 0.60)]]          # the submitted 9771150 point
    for bi, b in enumerate(BASES):
        print(f"\n{'='*104}\n=== BASE {b}  (plain 0.5 thresh: " +
              "  ".join(f"{REG[r]} n_cc={R[bi,r,0]} n_small={R[bi,r,1]} vox={R[bi,r,2]}"
                        for r in range(3)) + ")")
        print(f"{'lo':>5}{'hi':>5} | " + " | ".join(f"{r}:  n_cc n_small      vox  dvox%" for r in REG))
        for hi in HIS:
            for lo in LOS:
                a = A[KIDX[(b, lo, hi)]]
                cells = []
                for r in range(3):
                    dv = 100.0 * (a[r, 2] - b0[r, 2]) / max(b0[r, 2], 1)
                    cells.append(f"{a[r,0]:6d} {a[r,1]:6d} {a[r,2]:9d} {dv:+6.1f}")
                tag = "  <== SUBMITTED 9771150" if (b, lo, hi) == ("topk_comp", 0.40, 0.60) else ""
                print(f"{lo:5.2f}{hi:5.2f} | " + " | ".join(cells) + tag)

    print("\n=== instance-count delta vs submitted 9771150  (F1 is the binding axis) ===")
    print("    dcc>0 = more instances detected (F1 up) | dcc<0 = fewer (merged/dropped) = F1 risk")
    for b in BASES:
        for hi in HIS:
            for lo in LOS:
                a = A[KIDX[(b, lo, hi)]]
                d = [a[r, 0] - b0[r, 0] for r in range(3)]
                ds = [a[r, 1] - b0[r, 1] for r in range(3)]
                print(f"  {b:>12} lo={lo:.2f} hi={hi:.2f}   dcc WT/TC/ET {d[0]:+5d}/{d[1]:+5d}/{d[2]:+5d}"
                      f"    d_small {ds[0]:+5d}/{ds[1]:+5d}/{ds[2]:+5d}")
    print("PROBE_COMPLETE")
