"""How dilated are D015's ACTUAL predictions vs the 3-way? Decides the erosion.

build_d015 trained on targets where isolated small mets were dilated k=2, but we do
NOT know the net learned the full k=2 extent in 150 FT epochs. Measure it before eroding:
for TC small components present in BOTH decodes at the same location, compare sizes.
  ratio ~ (k=2 volume)/(true) ~ 4-7x  -> net learned it, erode k=2
  ratio ~ 1                          -> net did NOT dilate, DO NOT erode
"""
import os, glob
import numpy as np
from scipy.ndimage import label as cc_label, center_of_mass

V = os.path.expanduser("~/brats2025/val_infer")
D015 = f"{V}/pred_d015_s025"
REF = f"{V}/pred_topk_s025"        # a clean (undilated) member for size reference
LO, HI = 0.30, 0.70
STRUCT = np.ones((3, 3, 3), int)


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo, STRUCT)
    if n == 0:
        return np.zeros(p.shape, bool)
    cm = np.zeros(n + 1, np.float32)
    np.maximum.at(cm, lab.ravel(), p.ravel())
    keep = cm > hi
    keep[0] = False
    return keep[lab]


cases = sorted(os.path.basename(f)[:-4] for f in glob.glob(D015 + "/*.npz"))[:50]
d_small, r_small = [], []
matched = []
for cid in cases:
    dp = np.load(f"{D015}/{cid}.npz")["probabilities"]
    rp = np.load(f"{REF}/{cid}.npz")["probabilities"]
    dtc = hyst((dp[0] + dp[2]) / 2 if dp.shape[0] > 3 else dp[1], LO, HI)  # tc channel guard
    # 4-region channel order is [wt,tc,et,rc]; tc = channel 1
    dtc = hyst(dp[1], LO, HI)
    rtc = hyst(rp[1], LO, HI)
    dl, dn = cc_label(dtc, STRUCT)
    rl, rn = cc_label(rtc, STRUCT)
    for i in range(1, dn + 1):
        m = dl == i
        sz = int(m.sum())
        if sz < 1500:
            d_small.append(sz)
        # match to nearest ref component by centroid overlap
        c = tuple(int(round(x)) for x in center_of_mass(m))
        rid = rl[c] if rl[c] > 0 else 0
        if rid:
            rsz = int((rl == rid).sum())
            if sz < 1500 and rsz < 1500:
                matched.append((sz, rsz))
    for j in range(1, rn + 1):
        rsz = int((rl == j).sum())
        if rsz < 1500:
            r_small.append(rsz)

d_small = np.array(d_small)
r_small = np.array(r_small)
print(f"D015: {len(d_small)} small TC comps  median {np.median(d_small):.0f}  mean {d_small.mean():.0f}")
print(f"REF : {len(r_small)} small TC comps  median {np.median(r_small):.0f}  mean {r_small.mean():.0f}")
if matched:
    m = np.array(matched, float)
    ratio = m[:, 0] / np.maximum(m[:, 1], 1)
    print(f"MATCHED comps n={len(m)}  median D015/REF size ratio = {np.median(ratio):.2f}  "
          f"(mean {ratio.mean():.2f})")
    print("READ: ratio >> 1 (say >2) => D015 learned the dilation, ERODE. ~1 => do NOT erode.")
