#!/usr/bin/env python
"""Convert BraTS-MET 2025 -> nnU-Netv2 Dataset501_BraTSMET.
Channel order: 0000=t1n 0001=t1c 0002=t2w 0003=t2f. Labels: 1=NCR 2=ED 3=ET."""
import os, json, shutil, glob, sys
from pathlib import Path

RAW = Path.home()/'brats2025'/'extracted'
OUT = Path.home()/'brats2025'/'nnUNet_raw'/'Dataset501_BraTSMET'
CORR = Path.home()/'brats2025'/'extracted'/'MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels'
CH = ['t1n','t1c','t2w','t2f']  # -> 0000..0003

def main():
    tr_dir = next(RAW.glob('*Training*'))
    imagesTr = OUT/'imagesTr'; labelsTr = OUT/'labelsTr'
    imagesTr.mkdir(parents=True, exist_ok=True); labelsTr.mkdir(parents=True, exist_ok=True)
    # corrected seg lookup: BraTS-MET-XXXXX-YYY-seg.nii.gz
    corr = {p.name.replace('-seg.nii.gz',''): p for p in CORR.glob('*-seg.nii.gz')} if CORR.exists() else {}
    cases = sorted([d for d in tr_dir.iterdir() if d.is_dir()])
    n=0; applied=[]
    for c in cases:
        cid = c.name
        for i,m in enumerate(CH):
            src = c/f'{cid}-{m}.nii.gz'
            if not src.exists(): raise FileNotFoundError(src)
            os.symlink(src, imagesTr/f'{cid}_{i:04d}.nii.gz')
        seg = corr.get(cid, c/f'{cid}-seg.nii.gz')
        if cid in corr: applied.append(cid)
        os.symlink(seg, labelsTr/f'{cid}.nii.gz')
        n+=1
    dj = {
        'channel_names': {'0':'t1n','1':'t1c','2':'t2w','3':'t2f'},
        'labels': {'background':0, 'NCR':1, 'ED':2, 'ET':3},
        'numTraining': n, 'file_ending':'.nii.gz',
        'name':'BraTSMET2025','description':'BraTS 2025 Brain Metastasis Task1'
    }
    json.dump(dj, open(OUT/'dataset.json','w'), indent=2)
    print(f'cases={n} corrected_applied={applied}')
    print('dataset.json ->', OUT/'dataset.json')

if __name__=='__main__': main()
