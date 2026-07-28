#!/usr/bin/env python3
"""BraTS-METS inference entrypoint for the Synapse model-to-data container.

Reads test cases from --input-dir, runs the trained nnU-Net ResEnc-L model
(Dataset007, region-based WT/TC/ET/RC, fold 'all', checkpoint_final), applies
conservative connected-component postprocessing, and writes one segmentation
NIfTI per case to --output-dir.

Channel order (must match the trained dataset.json):
    0000 = t1c   0001 = t1n   0002 = t2f   0003 = t2w
Output labels: 0 bg, 1 NETC, 2 SNFH, 3 ET, 4 RC (nnU-Net applies
regions_class_order automatically when converting the region logits).

NOTE: the exact BraTS-METS container I/O contract (per-case folder layout and
output filename) should be confirmed against the official challenge packaging
spec. This script auto-detects the two common layouts and is easy to adjust at
the CASE-DISCOVERY and OUTPUT-NAMING blocks below.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

# channel suffix -> nnU-Net channel index
MODALITY_ORDER = ["t1c", "t1n", "t2f", "t2w"]
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/opt/nnunet/results/Dataset007_BraTSMets2025/"
    "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
)
# Conservative small-component removal. BraTS lesion-wise scoring EXCLUDES
# lesions < ~2 mm^3 and penalizes false negatives hard, so keep this gentle:
# only strip specks below MIN_COMPONENT_VOXELS. Tune on a held-out BraTS set.
MIN_COMPONENT_VOXELS = int(os.environ.get("MIN_COMPONENT_VOXELS", "5"))


def find_cases(input_dir: Path):
    """Return {case_id: {modality: path}}. Supports:
    (a) one sub-folder per case containing <id>-<mod>.nii.gz   (standard BraTS)
    (b) flat directory of <id>-<mod>.nii.gz files
    """
    cases = {}

    def register(case_id, mod, path):
        cases.setdefault(case_id, {})[mod] = path

    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    search_roots = subdirs if subdirs else [input_dir]
    for root in search_roots:
        for f in root.glob("*.nii.gz"):
            name = f.name[:-7]  # strip .nii.gz
            for mod in MODALITY_ORDER:
                if name.endswith(f"-{mod}") or name.endswith(f"_{mod}"):
                    case_id = name[: -(len(mod) + 1)]
                    register(case_id, mod, f)
                    break
    return cases


def stage_nnunet_input(cases, staging: Path):
    """Symlink/copy modalities into nnU-Net's expected <case>_000X.nii.gz form."""
    staging.mkdir(parents=True, exist_ok=True)
    ok = []
    for case_id, mods in sorted(cases.items()):
        if not all(m in mods for m in MODALITY_ORDER):
            missing = [m for m in MODALITY_ORDER if m not in mods]
            print(f"[warn] {case_id} missing {missing} — skipping", file=sys.stderr)
            continue
        for idx, mod in enumerate(MODALITY_ORDER):
            dst = staging / f"{case_id}_{idx:04d}.nii.gz"
            try:
                os.symlink(mods[mod], dst)
            except OSError:
                shutil.copy2(mods[mod], dst)
        ok.append(case_id)
    return ok


def postprocess(seg_path: Path):
    """Remove tiny specks per foreground label (conservative)."""
    if MIN_COMPONENT_VOXELS <= 1:
        return
    import cc3d

    img = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(img).astype(np.uint8)
    for label in (1, 2, 3, 4):
        mask = arr == label
        if not mask.any():
            continue
        cc = cc3d.connected_components(mask, connectivity=26)
        for cid in range(1, cc.max() + 1):
            comp = cc == cid
            if comp.sum() < MIN_COMPONENT_VOXELS:
                arr[comp] = 0
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(seg_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/input")
    ap.add_argument("--output-dir", default="/output")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(input_dir)
    if not cases:
        print(f"[error] no cases found under {input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] discovered {len(cases)} case(s)")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,        # test-time augmentation
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_DIR,
        use_folds=("all",),
        checkpoint_name="checkpoint_final.pth",
    )

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "in"
        nn_out = Path(tmp) / "out"
        nn_out.mkdir(parents=True, exist_ok=True)
        ok_cases = stage_nnunet_input(cases, staging)
        if not ok_cases:
            print("[error] no complete cases to predict", file=sys.stderr)
            sys.exit(1)

        predictor.predict_from_files(
            str(staging),
            str(nn_out),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )

        # OUTPUT NAMING: nnU-Net writes <case_id>.nii.gz. BraTS typically expects
        # <case_id>.nii.gz (or <case_id>-seg.nii.gz). Adjust suffix here if needed.
        for case_id in ok_cases:
            src = nn_out / f"{case_id}.nii.gz"
            if not src.exists():
                print(f"[warn] no prediction for {case_id}", file=sys.stderr)
                continue
            postprocess(src)
            shutil.copy2(src, output_dir / f"{case_id}.nii.gz")
            print(f"[ok] {case_id}")

    print("[done]")


if __name__ == "__main__":
    main()
