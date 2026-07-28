"""v1 and v2 are the SAME model at two points on a recall/precision dial.

  v1 (no precision term)   TP 2.40  FP 1.66   <- finds more, invents more
  v2 (lambda_fp = 0.2)     TP 1.82  FP 0.79   <- invents less, finds less

Neither wins. But they are one architecture, one dataset, one knob -- so the space
BETWEEN them is reachable for free, two ways:

  BLEND: blob member = a*v1 + (1-a)*v2, then 3-way with topk + completion.
         A continuous recall/precision dial, no GPU.

  VETO : decode with v1 (high recall), then DROP any component v2 does not vouch for
         (v2's PEAK probability inside it < t). This is NOT the size/confidence filter
         that failed four times -- those used hand-picked heuristics that correlate with
         being a small REAL lesion. v2 is a network EXPLICITLY TRAINED to recognise a
         false positive. It is a learned veto.

Target: beat the staged slot-1 submission (3way-v1 .30/.70) on all six tumour metrics:
        DSC .688/.720/.702   F1 .424/.483/.472   FP 0.85
"""
import os, sys, json, tempfile
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import label as cc_label

HOME = os.path.expanduser("~")
LE = f"{HOME}/brats2025/localeval"
CFG = f"{HOME}/brats2025/brats_eval/brats_evaluation/configs/config_mets.yaml"
VOL_THRESH, OVERLAP_THRESH = 20.0, 0.1
REGIONS = ["wt", "tc", "et"]

# (name, kind, alpha, lo, hi, veto)
V = [("REF 3way-v1 .30/.70", "blend", 1.00, 0.30, 0.70, None)]   # the staged slot-1 sub
for a in (0.75, 0.60, 0.50, 0.40, 0.25, 0.00):
    V.append((f"blend a={a:.2f} .30/.70", "blend", a, 0.30, 0.70, None))
for a in (0.75, 0.60, 0.50):
    V.append((f"blend a={a:.2f} .30/.65", "blend", a, 0.30, 0.65, None))
for t in (0.20, 0.30, 0.40, 0.50, 0.60):
    V.append((f"VETO v2>{t:.2f} .30/.70", "veto", 1.00, 0.30, 0.70, t))
VARIANTS = V


def bases(cid):
    pt = np.load(f"{LE}/pred_topk/{cid}.npz")["probabilities"][:3]
    cp = np.load(f"{LE}/pred_completion/{cid}.npz")["probabilities"][:3]
    c1 = np.load(f"{LE}/pred_ccloss/{cid}.npz")["probabilities"][:3]
    c2 = np.load(f"{LE}/pred_ccloss_v2/{cid}.npz")["probabilities"][:3]
    return pt, cp, c1, c2


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool), None, 0
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab], lab, n


def decode(pt, cp, c1, c2, kind, a, lo, hi, veto):
    if kind == "blend":
        blob = a * c1 + (1 - a) * c2
        avg = (pt + cp + blob) / 3.0
        seg = np.zeros(avg.shape[1:], np.uint8)
        for ch, cls in zip([0, 1, 2], [2, 1, 3]):
            m, _, _ = hyst(avg[ch], lo, hi)
            seg[m] = cls
        return seg
    # VETO: decode with v1, then drop components v2 does not vouch for
    avg = (pt + cp + c1) / 3.0
    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):
        m, _, _ = hyst(avg[ch], lo, hi)
        if not m.any():
            continue
        lab, n = cc_label(m)
        if n:
            vmax = np.zeros(n + 1, np.float32)
            np.maximum.at(vmax, lab.ravel(), c2[ch].ravel())   # v2's PEAK inside each component
            keep = vmax >= veto
            keep[0] = False
            m = keep[lab]
        seg[m] = cls
    return seg


def parse_region(d):
    num_fp = d.get("fp", 0) or 0
    refs = d.get("reference_instances", []) or []
    ld, ln = [], []
    tp = fn = 0
    small = large = False
    for r in refs:
        v = r.get("volume")
        if v is None:
            continue
        is_l = v >= VOL_THRESH
        large |= is_l
        small |= not is_l
        mt = r.get("is_matched") == 1
        sq = r.get("sq_dsc")
        if mt and is_l:
            ld.append(sq); ln.append(r.get("sq_nsd"))
        elif mt and not is_l:
            if sq is not None and sq >= OVERLAP_THRESH:
                tp += 1
            else:
                fn += 1
        elif (not mt) and is_l:
            ld.append(0); ln.append(0)
        else:
            fn += 1
    if num_fp:
        ld.extend([0] * num_fp); ln.extend([0] * num_fp)
    o = {}
    if d.get("n_ref_instances", 0) == 0 or not large:
        o["dsc"] = o["nsd"] = np.nan
    else:
        o["dsc"] = float(np.mean(ld)) if ld else 0.0
        nz = [x for x in ln if x is not None]
        o["nsd"] = float(np.mean(nz)) if nz else 0.0
    if small:
        den = 2 * tp + num_fp + fn
        o["f1"] = (2 * tp / den) if den else 0.0
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
    ev = _ev()
    pt, cp, c1, c2 = bases(cid)
    gt = f"{LE}/gt/{cid}.nii.gz"
    ref = nib.load(gt)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, kind, a, lo, hi, veto in VARIANTS:
            seg = decode(pt, cp, c1, c2, kind, a, lo, hi, veto)
            p = f"{td}/p.nii.gz"
            nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header), p)
            try:
                g2r = ev.evaluate(p, gt)
                out[name] = {r: parse_region(g2r[r].to_dict(True)) for r in REGIONS if r in g2r}
            except Exception:
                out[name] = {r: {k: np.nan for k in ("dsc", "nsd", "f1", "tp", "fn", "fp")}
                             for r in REGIONS}
    return cid, out


if __name__ == "__main__":
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_ccloss_v2/{c}.npz")]
    print(f"[blend] {len(cases)} cases x {len(VARIANTS)} variants", flush=True)
    KEYS = ["dsc", "nsd", "f1", "tp", "fn", "fp"]
    acc = {v[0]: {r: {k: [] for k in KEYS} for r in REGIONS} for v in VARIANTS}
    with Pool(NPROC) as pool:
        for i, (cid, res) in enumerate(pool.imap_unordered(one_case, cases)):
            for vn, per in res.items():
                for r, m in per.items():
                    for k in KEYS:
                        acc[vn][r][k].append(m.get(k, np.nan))
            if (i + 1) % 20 == 0:
                print(f"  ..{i+1}/{len(cases)}", flush=True)

    def nm(xs):
        xs = [x for x in xs if x is not None]
        return float(np.nanmean(xs)) if len(xs) else float("nan")
    fmt = lambda xs: "/".join(f"{x:.3f}" for x in xs)

    rows = {}
    hdr = ("variant".ljust(22) + " | " + "DSC WT/TC/ET".center(17) + " | " +
           "NSD WT/TC/ET".center(17) + " | " + "F1 WT/TC/ET".center(17) + " |    TP    FP    FN")
    print("\n" + hdr); print("-" * len(hdr))
    for name, *_ in VARIANTS:
        d = [nm(acc[name][r]["dsc"]) for r in REGIONS]
        s = [nm(acc[name][r]["nsd"]) for r in REGIONS]
        f = [nm(acc[name][r]["f1"]) for r in REGIONS]
        tp, fp, fn = (nm(acc[name]["wt"][k]) for k in ("tp", "fp", "fn"))
        rows[name] = {"dsc": d, "nsd": s, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        print(f"{name:<22} | {fmt(d)} | {fmt(s)} | {fmt(f)} | {tp:5.2f} {fp:5.2f} {fn:5.2f}")
    json.dump(rows, open(f"{LE}/blend_scores.json", "w"), indent=1)

    ref = rows["REF 3way-v1 .30/.70"]
    print("\n=== BEATS THE STAGED SLOT-1 SUBMISSION ON ALL SIX? ===")
    print(f"  anchor  DSC {fmt(ref['dsc'])}  F1 {fmt(ref['f1'])}  FP {ref['fp']:.2f}")
    win = []
    for n, r in rows.items():
        if n.startswith("REF"):
            continue
        if (all(r["dsc"][i] >= ref["dsc"][i] for i in range(3)) and
                all(r["f1"][i] >= ref["f1"][i] for i in range(3))):
            gain = sum(r["dsc"]) + sum(r["f1"]) - sum(ref["dsc"]) - sum(ref["f1"])
            win.append((gain, n, r))
    win.sort(reverse=True)
    if win:
        for g, n, r in win:
            print(f"  +{g:.3f}  {n:<22} DSC {fmt(r['dsc'])}  F1 {fmt(r['f1'])}  FP {r['fp']:.2f}  TP {r['tp']:.2f}")
    else:
        print("  (none) — the staged 3-way with v1 stands.")
    print("SCORE_BLEND_COMPLETE")
