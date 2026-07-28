#!/usr/bin/env python3
"""Build Dataset007_BraTSMets2025 (nnU-Netv2 raw) — challenge submission.
 - ALL 1296 cases -> imagesTr/labelsTr (train on whole dataset, no split).
 - channels: 0=t1c 1=t1n 2=t2f 3=t2w ; BraTS-MET-XXXXX-YYY -> sub-XXXXX-YYY
 - REGION-BASED labels: WT=1+2+3, TC=1+3, ET=3, RC=4 ; regions_class_order=[2,1,3,4]
 - applies the 2 corrected segs. Stdlib only."""
import json, shutil
from pathlib import Path
HOME=Path.home()
SOURCE=HOME/'brats2025'/'extracted'/'MICCAI-LH-BraTS2025-MET-Challenge-Training'
CORR=HOME/'brats2025'/'extracted'/'MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels'
OUT=HOME/'projects'/'CAI4CAI'/'SegData'/'raw'/'Dataset007_BraTSMets2025'
UCSD='UCSD - Training'
MOD={'t1c':0,'t1n':1,'t2f':2,'t2w':3}

def discover():
    cs=[]
    for d in sorted(SOURCE.iterdir()):
        if d.is_dir() and d.name.startswith('BraTS-MET-') and d.name!=UCSD: cs.append(d)
    u=SOURCE/UCSD
    if u.exists():
        for d in sorted(u.iterdir()):
            if d.is_dir() and d.name.startswith('BraTS-MET-'): cs.append(d)
    return cs

def tgt(n): return 'sub-'+n.replace('BraTS-MET-','')

def main():
    assert SOURCE.exists(), SOURCE
    (OUT/'imagesTr').mkdir(parents=True,exist_ok=True); (OUT/'labelsTr').mkdir(parents=True,exist_ok=True)
    corr={p.name.replace('-seg.nii.gz',''):p for p in CORR.glob('*-seg.nii.gz')} if CORR.exists() else {}
    cases=discover(); applied=[]
    for d in cases:
        cid=d.name; t=tgt(cid)
        for m,ch in MOD.items():
            s=d/f'{cid}-{m}.nii.gz'
            assert s.exists(), s
            shutil.copy2(s, OUT/'imagesTr'/f'{t}_{ch:04d}.nii.gz')
        seg=corr.get(cid, d/f'{cid}-seg.nii.gz')
        assert seg.exists(), seg
        shutil.copy2(seg, OUT/'labelsTr'/f'{t}.nii.gz')
        if cid in corr: applied.append(t)
    dj={'channel_names':{'0':'T1c','1':'T1n','2':'T2f','3':'T2w'},
        'labels':{'background':0,'whole_tumor':[1,2,3],'tumor_core':[1,3],
                  'enhancing_tumor':[3],'resection_cavity':[4]},
        'regions_class_order':[2,1,3,4],
        'numTraining':len(cases),'file_ending':'.nii.gz',
        'description':'BraTS2025 Brain Metastases, region-based. WT=1+2+3,TC=1+3,ET=3,RC=4. All cases, corrected labels applied.',
        'licence':'CC-BY-SA 4.0','release':'2.0-corrected-regionbased'}
    json.dump(dj, open(OUT/'dataset.json','w'), indent=2)
    json.dump({'total':len(cases),'corrected_applied':applied,'source':str(SOURCE)},
              open(OUT/'split_metadata.json','w'), indent=2)
    print('cases',len(cases),'corrected',applied)
    print('imagesTr',len(list((OUT/'imagesTr').glob('*.nii.gz'))),
          'labelsTr',len(list((OUT/'labelsTr').glob('*.nii.gz'))))
if __name__=='__main__': main()
