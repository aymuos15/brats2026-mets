#!/bin/bash
# Run AFTER the step-0.125 re-prediction finishes. Assembles + verifies + zips the
# RC-step probe (tumour byte-identical to 9771992). Does NOT submit (slot cost).
set -u
cd ~/brats2025
D=val_infer/pred_ccloss_s0125
if [ ! -f "$D/.done" ]; then
  echo "NOT READY: $D/.done missing ($(ls $D/*.npz 2>/dev/null | wc -l)/179 npz predicted)"; exit 1
fi
echo "=== assemble RC-lowstep (tumour=9771992 frozen, RC=step0.125) ==="
./evalenv/bin/python assemble_rc_lowstep.py sub_rc_s0125 8 || { echo ASSEMBLE_FAILED; exit 1; }
N=$(ls val_infer/sub_rc_s0125/*.nii.gz 2>/dev/null | wc -l)
[ "$N" = "179" ] || { echo "GATE FAIL: $N/179 masks"; exit 1; }
( cd val_infer/sub_rc_s0125 && zip -q -j ~/brats2025/sub_rc_s0125.zip *.nii.gz )
echo "ZIPPED sub_rc_s0125.zip ($N masks): $(unzip -l ~/brats2025/sub_rc_s0125.zip | tail -1)"
echo "TO SUBMIT (consumes a 07-19 slot): ./.venv/bin/python submit.py sub_rc_s0125.zip rc_step0125"
echo "FINISH_RC_LOWSTEP_DONE"
