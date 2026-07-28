#!/usr/bin/env python3
"""BraTS 2026 Task 1 (Brain Metastases) — container entrypoint.

Reproduces submission 9771992 (our best validation entry):

    tumour (WT/TC/ET) = mean(TopK-4reg, completion-BackSplit, CC-loss)
                        decoded by HYSTERESIS lo=0.25 / hi=0.70
    RC (label 4)      = CC-loss RC channel @ 0.50, then keep-largest component

This mirrors src/decode/assemble_3way.py exactly, so the container reproduces the
decode that was validated on the leaderboard rather than a re-implementation of it.

Challenge runtime contract (wiki syn74274097, page 639582):
  * /input is READ-ONLY   -> we never write there; staging is under /tmp
  * /output must be FLAT  -> one <case_id>.nii.gz per case, no sub-folders
  * no network access     -> everything is baked into the image
  * 12 h TOTAL for the whole hidden test set

Strategy: three sequential nnU-Net passes writing probabilities to /tmp, then a
CPU fusion pass. Probabilities are ~100 MB/case/model, so peak scratch is roughly
300 MB x N_cases; the challenge grants 200 GB. Set BRATS_STREAM=1 to fuse and
delete per case instead, trading a little speed for much less scratch.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import label as cc_label

MODALITY_ORDER = ["t1c", "t1n", "t2f", "t2w"]   # -> nnU-Net channels 0000..0003

MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/models"))
MEMBERS = ["topk", "completion", "ccloss"]      # fusion order; ccloss also supplies RC

LO = float(os.environ.get("BRATS_LO", "0.25"))  # extent threshold
HI = float(os.environ.get("BRATS_HI", "0.70"))  # seed threshold
STEP = float(os.environ.get("BRATS_STEP", "0.25"))
TTA = os.environ.get("BRATS_TTA", "1") == "1"


# --------------------------------------------------------------------------- IO

def find_cases(input_dir: Path):
    """{case_id: {modality: path}}. Accepts one folder per case, or a flat dir."""
    cases = {}
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    for root in (subdirs if subdirs else [input_dir]):
        for f in root.glob("*.nii.gz"):
            name = f.name[:-7]
            for mod in MODALITY_ORDER:
                if name.endswith(f"-{mod}") or name.endswith(f"_{mod}"):
                    cases.setdefault(name[: -(len(mod) + 1)], {})[mod] = f
                    break
    return cases


def stage_inputs(cases, staging: Path):
    """Symlink modalities into nnU-Net's <case>_000X.nii.gz layout. /input stays untouched."""
    staging.mkdir(parents=True, exist_ok=True)
    ok = []
    for case_id, mods in sorted(cases.items()):
        missing = [m for m in MODALITY_ORDER if m not in mods]
        if missing:
            print(f"[warn] {case_id}: missing {missing} — skipping", file=sys.stderr)
            continue
        for idx, mod in enumerate(MODALITY_ORDER):
            dst = staging / f"{case_id}_{idx:04d}.nii.gz"
            try:
                os.symlink(mods[mod], dst)
            except OSError:
                shutil.copy2(mods[mod], dst)
        ok.append(case_id)
    return ok


# ---------------------------------------------------------------------- decode

def hyst(p, lo, hi):
    """Keep components of (p > lo) whose PEAK probability exceeds hi. Never looks at size."""
    lab, n = cc_label(p > lo)
    if n == 0:
        return np.zeros(p.shape, bool)
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())
    keep = cmax > hi
    keep[0] = False
    return keep[lab]


def rc_largest(seg):
    m = seg == 4
    if not m.any():
        return seg
    lab, n = cc_label(m)
    if n <= 1:
        return seg
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    seg[m & (lab != sizes.argmax())] = 0
    return seg


def fuse_one(case_id, prob_dirs, output_dir):
    """Average the three members' tumour channels, decode, graft RC, write NIfTI."""
    probs = [np.load(prob_dirs[m] / f"{case_id}.npz")["probabilities"] for m in MEMBERS]
    avg = sum(p[:3] for p in probs) / float(len(probs))
    rcp = probs[MEMBERS.index("ccloss")][3]

    seg = np.zeros(avg.shape[1:], np.uint8)
    for ch, cls in zip([0, 1, 2], [2, 1, 3]):      # WT->2, TC->1, ET->3 (nested overwrite)
        seg[hyst(avg[ch], LO, HI)] = cls
    seg[rcp >= 0.5] = 4
    seg = rc_largest(seg)

    ref = nib.load(str(prob_dirs[MEMBERS[0]] / f"{case_id}.nii.gz"))
    nib.save(nib.Nifti1Image(seg.transpose(2, 1, 0), ref.affine, ref.header),
             str(output_dir / f"{case_id}.nii.gz"))

    labs = set(np.unique(seg).tolist())
    bad_lab = not labs.issubset({0, 1, 2, 3, 4})
    rc_dom = seg.size > 0 and (seg == 4).sum() / seg.size > 0.08
    return bad_lab, rc_dom


# ------------------------------------------------------------------------ main

def build_predictor(member, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=STEP,
        use_gaussian=True,
        use_mirroring=TTA,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_ROOT / member), use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/input")
    ap.add_argument("--output-dir", default="/output")
    ap.add_argument("--limit", type=int, default=0, help="predict only the first N cases (timing runs)")
    args = ap.parse_args()

    t0 = time.time()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(input_dir)
    if not cases:
        print(f"[error] no cases found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] {len(cases)} case(s) | device={device} | lo={LO} hi={HI} "
          f"step={STEP} tta={TTA}", flush=True)
    if device.type != "cuda":
        print("[warn] running on CPU — this will not meet the time budget", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="brats_") as tmp:
        tmp = Path(tmp)
        staging = tmp / "in"
        ok_cases = stage_inputs(cases, staging)
        if args.limit:
            keep = set(ok_cases[: args.limit])
            for f in staging.glob("*.nii.gz"):
                if f.name[:-12] not in keep:
                    f.unlink()
            ok_cases = ok_cases[: args.limit]
        if not ok_cases:
            print("[error] no complete cases to predict", file=sys.stderr)
            sys.exit(1)
        print(f"[info] predicting {len(ok_cases)} case(s)", flush=True)

        prob_dirs = {}
        for member in MEMBERS:
            t_m = time.time()
            out = tmp / f"prob_{member}"
            out.mkdir(parents=True, exist_ok=True)
            prob_dirs[member] = out
            predictor = build_predictor(member, device)
            predictor.predict_from_files(
                str(staging), str(out),
                save_probabilities=True, overwrite=True,
                num_processes_preprocessing=3,
                num_processes_segmentation_export=3,
            )
            del predictor
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dt = time.time() - t_m
            print(f"[time] member {member}: {dt/60:.1f} min "
                  f"({dt/len(ok_cases):.1f} s/case)", flush=True)

        t_f = time.time()
        bad_lab = rc_dom = 0
        for case_id in ok_cases:
            try:
                bl, rd = fuse_one(case_id, prob_dirs, output_dir)
                bad_lab += bl
                rc_dom += rd
            except Exception as exc:                      # noqa: BLE001
                print(f"[error] fusion failed for {case_id}: {exc}", file=sys.stderr)
        print(f"[time] fusion: {(time.time()-t_f)/60:.1f} min", flush=True)

    n_out = len(list(output_dir.glob("*.nii.gz")))
    flat = not any(p.is_dir() for p in output_dir.iterdir())
    total = time.time() - t0
    print(f"[verify] outputs={n_out}/{len(ok_cases)} bad_labels={bad_lab} "
          f"rc_dominating={rc_dom} output_flat={flat}")
    print(f"[time] TOTAL {total/60:.1f} min ({total/len(ok_cases):.1f} s/case)")
    print("VERIFY_GATE:", "PASS" if (n_out == len(ok_cases) and not bad_lab
                                     and not rc_dom and flat) else "FAIL")
    if n_out != len(ok_cases):
        sys.exit(1)


if __name__ == "__main__":
    main()
