"""(§1b) QUANTIFY THE MIMIC READ. Measure distance-from-brain-surface for
3-way TC FALSE-POSITIVE components vs GT TC lesion components, on the 80-case
local set. If FPs sit systematically nearer the brain surface than real mets,
the mimic read (§1c) is a real, MEASURABLE FP-separation axis (anatomical, not
confidence). If the two distributions overlap, the read is wrong and a surface
filter would kill real peripheral mets (grey-white-junction mets ARE peripheral
-- this is the confound we are checking, not assuming).

FP definition is byte-identical to fp_crops.py: a TC-decode component
(hysteresis 0.30/0.70 over the mean of topk+completion+ccloss) with ZERO GT
overlap. Brain mask = T1c>0 (BraTS is skull-stripped), holes filled. Surface
distance = EDT of the brain interior (1mm isotropic -> voxels == mm). Per
component we take the MEDIAN EDT over its voxels (how deep the bulk sits),
which is more robust than a single centroid. Split by size bucket so a
size confound (small mets happen to be peripheral) is visible.
"""
import os, json
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label, binary_fill_holes, distance_transform_edt

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
OUT = f"{HOME}/brats2025/fp_surface_dist.json"
STRUCT = np.ones((3, 3, 3), int)
LO, HI = 0.30, 0.70
TC_LABS = [1, 3]


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


def comp_records(mask, edt, p=None):
    """Per connected component: median surface-dist, size, (opt) peak prob."""
    lab, n = cc_label(mask, STRUCT)
    recs = []
    for i in range(1, n + 1):
        m = lab == i
        rec = {"size": int(m.sum()), "surf_med": float(np.median(edt[m])),
               "surf_min": float(edt[m].min()), "surf_max": float(edt[m].max())}
        if p is not None:
            rec["peak"] = float(p[m].max())
        recs.append((m, rec))
    return recs


def q(a):
    a = np.asarray(a, float)
    return dict(n=int(a.size), med=float(np.median(a)),
                q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)),
                mean=float(a.mean()))


if __name__ == "__main__":
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss/{c}.npz")]
    print(f"[surf] {len(cases)} cases", flush=True)
    fp_dist, fp_size = [], []
    gt_dist, gt_size = [], []
    fp_small_dist, gt_small_dist = [], []      # size < 200 vox
    for i, cid in enumerate(cases):
        gt_arr = np.asarray(nib.load(f"{LE}/gt/{cid}.nii.gz").dataobj)
        gt = np.isin(gt_arr, TC_LABS)
        t1c = np.asarray(nib.load(f"{LE}/images/{cid}_0000.nii.gz").dataobj)
        brain = binary_fill_holes(t1c > 0)
        edt = distance_transform_edt(brain)

        p = probs(cid)[1]
        pred = hyst(p, LO, HI)

        for m, rec in comp_records(pred, edt, p):
            if gt[m].any():
                continue                        # matched -> not an FP
            fp_dist.append(rec["surf_med"]); fp_size.append(rec["size"])
            if rec["size"] < 200:
                fp_small_dist.append(rec["surf_med"])
        for m, rec in comp_records(gt, edt):
            gt_dist.append(rec["surf_med"]); gt_size.append(rec["size"])
            if rec["size"] < 200:
                gt_small_dist.append(rec["surf_med"])
        if (i + 1) % 20 == 0:
            print(f"  ..{i+1}/{len(cases)}  FP={len(fp_dist)} GT={len(gt_dist)}", flush=True)

    fp_dist = np.array(fp_dist); gt_dist = np.array(gt_dist)
    # separation summary
    gt_q25 = np.percentile(gt_dist, 25)
    frac_fp_below_gt_q25 = float((fp_dist < gt_q25).mean())
    # AUC = P(random FP more peripheral than random GT lesion) via rank-sum
    from itertools import product
    # exact AUC without scipy: mean over pairs of [fp<gt] + 0.5*[fp==gt]
    less = 0.0; total = fp_dist.size * gt_dist.size
    # vectorized
    order = np.sort(gt_dist)
    idx_lt = np.searchsorted(order, fp_dist, side="left")
    idx_le = np.searchsorted(order, fp_dist, side="right")
    less = float(idx_lt.sum()); eq = float((idx_le - idx_lt).sum())
    auc_fp_more_peripheral = (less + 0.5 * eq) / total

    summ = {
        "fp_all": q(fp_dist), "gt_all": q(gt_dist),
        "fp_small<200": q(fp_small_dist), "gt_small<200": q(gt_small_dist),
        "fp_size": q(fp_size), "gt_size": q(gt_size),
        "gt_q25_surfdist": float(gt_q25),
        "frac_FP_more_peripheral_than_GT_q25": frac_fp_below_gt_q25,
        "AUC_FP_more_peripheral_than_GT": float(auc_fp_more_peripheral),
    }
    json.dump(summ, open(OUT, "w"), indent=2)
    print("\n==== SURFACE-DISTANCE (voxels==mm, higher=deeper in brain) ====")
    for k in ["fp_all", "gt_all", "fp_small<200", "gt_small<200"]:
        s = summ[k]
        print(f"  {k:16s} n={s['n']:4d}  med={s['med']:5.1f}  "
              f"IQR[{s['q25']:5.1f},{s['q75']:5.1f}]  mean={s['mean']:5.1f}")
    print(f"\n  GT lesion 25th-pctile surface-dist = {gt_q25:.1f} mm")
    print(f"  fraction of FPs more peripheral than that = {frac_fp_below_gt_q25:.2f}")
    print(f"  AUC(FP more peripheral than GT lesion)    = {auc_fp_more_peripheral:.3f}")
    print("\n  Read: AUC>>0.5 and non-overlapping IQRs => FPs ARE peripheral,")
    print("        real mets are not => mimic axis is measurable (build filter).")
    print("        AUC~0.5 or overlapping IQRs => read is WRONG, drop it.")
    print("SURF_DONE")
