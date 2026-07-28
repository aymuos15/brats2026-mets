"""Salvageable dilate variant (b): dilate ONLY small lesions that are ISOLATED (no other
lesion within 2k voxels), so dilation can never merge two instances. This measures the
ADDRESSABLE SET: of the small lesions, how many are (i) isolated at k=1/2 -> safe to dilate,
and crucially (ii) BLIND (3-way peak<0.10) -> the ones dilation might un-blind.

If the blind small lesions are mostly CLUSTERED (not isolated), isolated-only dilation can't
reach them and the variant is dead on arrival. If a good share of the blind ones are isolated,
the variant has a real, merge-free target.
"""
import os, json
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label, binary_dilation, generate_binary_structure

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
VOL_THRESH = 20.0
STRUCT = np.ones((3, 3, 3), int)
BALL = generate_binary_structure(3, 1)
REG = {"wt": (0, [1, 2, 3]), "tc": (1, [1, 3]), "et": (2, [3])}
KS = [1, 2]


def probs(cid):
    acc = None
    for d in ["pred_topk", "pred_completion", "pred_ccloss"]:
        p = np.transpose(np.load(f"{LE}/{d}/{cid}.npz")["probabilities"][:3], (0, 3, 2, 1))
        acc = p if acc is None else acc + p
    return acc / 3.0


def one(cid):
    img = nib.load(f"{LE}/gt/{cid}.nii.gz")
    a = np.asarray(img.dataobj)
    vox = float(np.prod(img.header.get_zooms()[:3]))
    P = probs(cid)
    out = {}
    for reg, (ch, labs) in REG.items():
        gt = np.isin(a, labs)
        glab, gn = cc_label(gt, STRUCT)
        p = P[ch]
        rows = []  # (small?, peak, {k: isolated})
        for i in range(1, gn + 1):
            m = glab == i
            if m.sum() * vox >= VOL_THRESH:
                continue
            peak = float(p[m].max())
            others = gt & ~m
            iso = {}
            for k in KS:
                iso[k] = not (binary_dilation(m, BALL, iterations=k) & others).any()
            rows.append((peak, iso))
        out[reg] = rows
    return out


if __name__ == "__main__":
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss/{c}.npz")]
    print(f"[iso] {len(cases)} cases | isolated & blind small lesions (merge-free dilation target)", flush=True)
    agg = {reg: {"n": 0, "blind": 0, "iso": {k: 0 for k in KS}, "iso_blind": {k: 0 for k in KS}} for reg in REG}
    with Pool(12) as pool:
        for o in pool.imap_unordered(one, cases):
            for reg in REG:
                for peak, iso in o[reg]:
                    agg[reg]["n"] += 1
                    blind = peak < 0.10
                    if blind:
                        agg[reg]["blind"] += 1
                    for k in KS:
                        if iso[k]:
                            agg[reg]["iso"][k] += 1
                            if blind:
                                agg[reg]["iso_blind"][k] += 1
    for reg in REG:
        d = agg[reg]
        print(f"\n=== {reg.upper()} — {d['n']} small lesions, {d['blind']} BLIND ({100*d['blind']/max(d['n'],1):.0f}%)")
        for k in KS:
            print(f"   k={k}: isolated {d['iso'][k]}/{d['n']} ({100*d['iso'][k]/max(d['n'],1):.0f}%)  |  "
                  f"ISOLATED & BLIND (addressable) {d['iso_blind'][k]}/{d['blind']} "
                  f"({100*d['iso_blind'][k]/max(d['blind'],1):.0f}% of blind)")
    print("\nAddressable = isolated&blind: small lesions dilation could un-blind WITHOUT merging any instance.")
    print("ISO_PRECHECK_COMPLETE")
