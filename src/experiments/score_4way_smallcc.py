"""Free ensembling test (user's 'why not fuse more models').

The only clean untried pure-average member is smallcc (D014): 80/80 coverage,
same 4-region (4,z,y,x) layout as topk/completion/ccloss. pred_ft is a 6-region
BackSplit model with 44/80 coverage -> unusable. So: does adding smallcc as a
4th equal-weight member beat the 3-way? Pure average -> component-count-safe ->
the harness is VALID here (§5). If §1's 'corroborated FP' finding holds, adding
a same-lineage member will NOT dilute the FPs and F1 will not move.

Reuses score_blend.py's decode/parse/panoptica pipeline verbatim so numbers are
directly comparable to the .7034/.4596 anchor. Tests both .30/.70 and the banked
.25/.70 extent so the comparison is against 9771992's real operating point too.
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

# (name, members, lo, hi)   members are dir keys, equal-weight averaged
VARIANTS = [
    ("REF 3way .30/.70",       ["topk", "completion", "ccloss"],            0.30, 0.70),
    ("REF 3way .25/.70",       ["topk", "completion", "ccloss"],            0.25, 0.70),
    ("4way+smallcc .30/.70",   ["topk", "completion", "ccloss", "smallcc"], 0.30, 0.70),
    ("4way+smallcc .25/.70",   ["topk", "completion", "ccloss", "smallcc"], 0.25, 0.70),
]


def load(cid, key):
    return np.load(f"{LE}/pred_{key}/{cid}.npz")["probabilities"][:3]


def hyst(p, lo, hi):
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab]


def decode(members_arrs, lo, hi):
    avg = sum(members_arrs) / len(members_arrs)
    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):
        seg[hyst(avg[ch], lo, hi)] = cls
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
    keys = sorted({k for _, ms, _, _ in VARIANTS for k in ms})
    arrs = {k: load(cid, k) for k in keys}
    gt = f"{LE}/gt/{cid}.nii.gz"
    ref = nib.load(gt)
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, ms, lo, hi in VARIANTS:
            seg = decode([arrs[k] for k in ms], lo, hi)
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
    NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cases = [c for c in json.load(open(f"{LE}/cases.json"))
             if os.path.exists(f"{LE}/pred_smallcc/{c}.npz")]
    print(f"[4way] {len(cases)} cases x {len(VARIANTS)} variants", flush=True)
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
    print("\n%-24s | %-17s | %-17s | %-17s |   TP    FP" %
          ("variant", "DSC WT/TC/ET", "NSD WT/TC/ET", "F1 WT/TC/ET"))
    print("-" * 92)
    for name, *_ in VARIANTS:
        d = [nm(acc[name][r]["dsc"]) for r in REGIONS]
        s = [nm(acc[name][r]["nsd"]) for r in REGIONS]
        f = [nm(acc[name][r]["f1"]) for r in REGIONS]
        tp, fp = nm(acc[name]["wt"]["tp"]), nm(acc[name]["wt"]["fp"])
        rows[name] = {"dsc": d, "nsd": s, "f1": f, "tp": tp, "fp": fp}
        print(f"{name:<24} | {fmt(d)} | {fmt(s)} | {fmt(f)} | {tp:5.2f} {fp:5.2f}")
    json.dump(rows, open(f"{LE}/fourway_smallcc_scores.json", "w"), indent=1)

    for anchor, cand in [("REF 3way .30/.70", "4way+smallcc .30/.70"),
                         ("REF 3way .25/.70", "4way+smallcc .25/.70")]:
        a, c = rows[anchor], rows[cand]
        df1 = [c["f1"][i] - a["f1"][i] for i in range(3)]
        ddsc = [c["dsc"][i] - a["dsc"][i] for i in range(3)]
        print(f"\n{cand} vs {anchor}:")
        print(f"  dF1  = {fmt(df1)}")
        print(f"  dDSC = {fmt(ddsc)}")
        better = all(c["f1"][i] >= a["f1"][i] for i in range(3)) and \
                 all(c["dsc"][i] >= a["dsc"][i] for i in range(3))
        print(f"  dominates on all 6 tumour cells? {better}")
    print("SCORE_4WAY_COMPLETE")
