"""(a) LOOK AT THE FALSE POSITIVES. Crop T1c around the 50 most confident FPs and render a
montage so a human can decide: mimic (vessel/choroid/dura -> fixable) or annotation gap
(real met the annotators missed -> not our ceiling to fix). We have run 7 suppression
experiments without ever looking at one.

FP = a TC-decode component (tumour core = NETC+ET) with ZERO overlap with any GT lesion,
in the 3-way ensemble we actually submit (hysteresis 0.30/0.70). Ranked by peak prob.
For each: axial T1c crop at the FP centroid with the FP boundary in red, plus T2-FLAIR crop.
"""
import os, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label, binary_dilation, center_of_mass, find_objects

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
OUT = f"{HOME}/brats2025/fp_montage.png"
STRUCT = np.ones((3, 3, 3), int)
LO, HI = 0.30, 0.70
TC_LABS = [1, 3]
CROP = 56          # half-size handled below; window = 2*R
R = 28
TOPN = 50


def probs(cid):
    acc = None
    for d in ["pred_topk", "pred_completion", "pred_ccloss"]:
        p = np.transpose(np.load(f"{LE}/{d}/{cid}.npz")["probabilities"][:3], (0, 3, 2, 1))
        acc = p if acc is None else acc + p
    return acc / 3.0


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo, STRUCT)
    if n == 0:
        return np.zeros(p.shape, bool)
    cm = np.zeros(n + 1, np.float32)
    np.maximum.at(cm, lab.ravel(), p.ravel())
    keep = cm > hi
    keep[0] = False
    return keep[lab]


def collect(cid):
    a = np.asarray(nib.load(f"{LE}/gt/{cid}.nii.gz").dataobj)
    gt = np.isin(a, TC_LABS)
    p = probs(cid)[1]                 # TC channel
    pred = hyst(p, LO, HI)
    plab, pn = cc_label(pred, STRUCT)
    fps = []
    for i in range(1, pn + 1):
        m = plab == i
        if gt[m].any():
            continue                  # overlaps GT -> not an FP
        peak = float(p[m].max())
        cz = int(round(center_of_mass(m)[2]))
        cx, cy = [int(round(c)) for c in center_of_mass(m.max(axis=2))]
        fps.append({"cid": cid, "peak": peak, "size": int(m.sum()),
                    "cx": cx, "cy": cy, "cz": cz})
    return fps


if __name__ == "__main__":
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss/{c}.npz")]
    print(f"[fp] scanning {len(cases)} cases for TC false positives", flush=True)
    allfp = []
    for i, c in enumerate(cases):
        allfp += collect(c)
        if (i + 1) % 20 == 0:
            print(f"  ..{i+1}/{len(cases)}  ({len(allfp)} FPs so far)", flush=True)
    allfp.sort(key=lambda d: -d["peak"])
    top = allfp[:TOPN]
    print(f"[fp] {len(allfp)} total FPs; rendering top {len(top)} by peak "
          f"(peak range {top[0]['peak']:.3f}..{top[-1]['peak']:.3f})", flush=True)

    ncol, nrow = 10, (len(top) + 9) // 10
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.0))
    axes = np.atleast_2d(axes)
    cache = {}
    for idx, fp in enumerate(top):
        ax = axes[idx // ncol][idx % ncol]
        cid = fp["cid"]
        if cid not in cache:
            t1c = np.asarray(nib.load(f"{LE}/images/{cid}_0000.nii.gz").dataobj).astype(np.float32)
            cache[cid] = t1c
        t1c = cache[cid]
        cx, cy, cz = fp["cx"], fp["cy"], fp["cz"]
        x0, x1 = max(0, cx - R), min(t1c.shape[0], cx + R)
        y0, y1 = max(0, cy - R), min(t1c.shape[1], cy + R)
        sl = t1c[x0:x1, y0:y1, cz]
        # window intensity for contrast
        v = sl[sl > 0]
        vmax = np.percentile(v, 99) if v.size else 1.0
        ax.imshow(sl.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        # mark FP centre
        ax.plot(cx - x0, cy - y0, "o", mfc="none", mec="red", ms=14, mew=1.5)
        ax.set_title(f"{cid.split('-')[-2]} p{fp['peak']:.2f} {fp['size']}v", fontsize=6)
        ax.axis("off")
    for j in range(len(top), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    plt.tight_layout()
    plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"[fp] wrote {OUT}", flush=True)
    # also dump the table for reference
    json.dump(top, open(f"{HOME}/brats2025/fp_top.json", "w"), indent=1)
    print("FP_CROPS_COMPLETE")
