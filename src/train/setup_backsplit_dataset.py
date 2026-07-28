#!/usr/bin/env python3
"""Set up the BackSplit variant of Dataset007 *without* re-preprocessing.

Idea (BackSplit, arXiv:2511.19394, applied to BraTS-METS):
  Sub-divide the structured background for the two weak targets (TC, ET) using the tumour
  envelope we already have. WT is the container of TC/ET (like the kidney contains the cyst in
  the paper). The disentangled shells are *already* raw BraTS classes:
        NETC (label 1) = TC \\ ET   (non-enhancing core)
        SNFH (label 2) = WT \\ TC   (peritumoral edema)
  We add them as two extra supervised region-channels alongside WT/TC/ET/RC. The network then
  gets direct, disentangled gradients for the structured background of TC and ET. At inference the
  4 real regions decode exactly as before (the aux channels write labels that are already correct),
  so the submission output is unchanged in format.

What this does:
  * Creates a new dataset id (default 11) whose preprocessed arrays are SYMLINKED to Dataset007's
    (no re-preprocess: region one-hot is computed on the fly from the stored integer label map).
  * Writes a new dataset.json with 6 regions + a redundant-safe regions_class_order.
  * Copies the ResEnc-L plans and sets 3d_fullres -> patch 96^3, batch 6, **batch_dice: true**
    (the sanity-check finding: stock default False is suboptimal for 96^3 small lesions).

Then train (no new trainer needed — reuse our best region loss):
  nnUNet_compile=0 nnUNetv2_train 11 3d_fullres all \\
      -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceTopK10BCE

VALIDATE ON-MACHINE before launch:
  - `nnUNetv2_train ... --val_disable_overwrite` dry init prints "num_segmentation_heads: 6".
  - confirm a couple of decoded val masks are byte-identical to the 4-region decode.
"""
import argparse, json, os, shutil

SRC_ID = "Dataset007_BraTSMets2025"
DST_ID_DEFAULT = "Dataset011_BraTSMetsBackSplit"

# 6 regions: 4 real (unchanged) + 2 BackSplit shells. Order matters for decode.
BACKSPLIT_LABELS = {
    "background": 0,
    "whole_tumor": [1, 2, 3],
    "tumor_core": [1, 3],
    "enhancing_tumor": [3],
    "resection_cavity": [4],
    "necrotic_core_aux": [1],   # NETC = TC \ ET  -> structured background shell for ET
    "edema_aux": [2],           # SNFH = WT \ TC  -> structured background shell for TC
}
# Real regions decode first (2,1,3,4 as before); aux regions write the SAME labels they overlap,
# so the final 4-class map is identical to the original 4-region decode.
BACKSPLIT_REGIONS_CLASS_ORDER = [2, 1, 3, 4, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--dst", default=DST_ID_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="actually create (default: dry run)")
    a = ap.parse_args()
    assert a.nnunet_preprocessed and a.nnunet_raw, "set nnUNet_preprocessed / nnUNet_raw"

    src_pp = os.path.join(a.nnunet_preprocessed, SRC_ID)
    dst_pp = os.path.join(a.nnunet_preprocessed, a.dst)
    print(f"src preprocessed: {src_pp}")
    print(f"dst preprocessed: {dst_pp}")
    assert os.path.isdir(src_pp), src_pp

    if not a.apply:
        print("\n[DRY RUN] would:")
        print(f"  mkdir {dst_pp}; symlink data dirs from {SRC_ID}")
        print(f"  write dataset.json with 6 regions: {list(BACKSPLIT_LABELS)}")
        print(f"  regions_class_order = {BACKSPLIT_REGIONS_CLASS_ORDER}")
        print("  copy nnUNetResEncUNetLPlans.json -> patch 96^3 / batch 6 / batch_dice=True")
        print("\nre-run with --apply to create.")
        return

    os.makedirs(dst_pp, exist_ok=True)
    # symlink the heavy data folders / fingerprint; copy small json we will edit
    for entry in os.listdir(src_pp):
        s = os.path.join(src_pp, entry)
        d = os.path.join(dst_pp, entry)
        if os.path.exists(d):
            continue
        if entry in ("dataset.json", "nnUNetResEncUNetLPlans.json"):
            continue  # handled below
        os.symlink(s, d)

    # dataset.json: start from source, override labels + regions_class_order
    with open(os.path.join(src_pp, "dataset.json")) as f:
        dj = json.load(f)
    dj["labels"] = BACKSPLIT_LABELS
    dj["regions_class_order"] = BACKSPLIT_REGIONS_CLASS_ORDER
    with open(os.path.join(dst_pp, "dataset.json"), "w") as f:
        json.dump(dj, f, indent=4)

    # plans: copy + edit 3d_fullres
    with open(os.path.join(src_pp, "nnUNetResEncUNetLPlans.json")) as f:
        plans = json.load(f)
    plans["dataset_name"] = a.dst
    c = plans["configurations"]["3d_fullres"]
    c["patch_size"] = [96, 96, 96]
    c["batch_size"] = 6
    c["batch_dice"] = True
    with open(os.path.join(dst_pp, "nnUNetResEncUNetLPlans.json"), "w") as f:
        json.dump(plans, f, indent=4)

    # mirror raw dataset.json so the raw dataset id resolves too
    src_raw = os.path.join(a.nnunet_raw, SRC_ID)
    dst_raw = os.path.join(a.nnunet_raw, a.dst)
    if os.path.isdir(src_raw) and not os.path.isdir(dst_raw):
        os.makedirs(dst_raw, exist_ok=True)
        for entry in os.listdir(src_raw):
            if entry == "dataset.json":
                continue
            os.symlink(os.path.join(src_raw, entry), os.path.join(dst_raw, entry))
        with open(os.path.join(dst_raw, "dataset.json"), "w") as f:
            json.dump(dj, f, indent=4)
    print("done. Train with: nnUNetv2_train 11 3d_fullres all -p nnUNetResEncUNetLPlans -tr nnUNetTrainerDiceTopK10BCE")


if __name__ == "__main__":
    main()
