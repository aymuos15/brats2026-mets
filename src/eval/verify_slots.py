import numpy as np, nibabel as nib, glob, os
V=os.path.expanduser("~/brats2025/val_infer")
ref=f"{V}/sub_3way_lo030_hi070"
s1=f"{V}/sub_slot1_rcens"
cases=sorted(os.path.basename(f)[:-7] for f in glob.glob(s1+"/*.nii.gz"))
badlab=big=rc1=rc_ref=tum_diff=rc_diff=0
for c in cases:
    a=np.asarray(nib.load(f"{s1}/{c}.nii.gz").dataobj).astype(np.uint8)
    if not set(np.unique(a).tolist()).issubset({0,1,2,3,4}): badlab+=1
    if a.size and max((a==l).sum() for l in (1,2,3,4))/a.size>0.08: big+=1
    rc1+=int((a==4).any())
    r=np.asarray(nib.load(f"{ref}/{c}.nii.gz").dataobj).astype(np.uint8)
    rc_ref+=int((r==4).any())
    if not np.array_equal(np.where(a==4,0,a),np.where(r==4,0,r)): tum_diff+=1
    if not np.array_equal(a==4,r==4): rc_diff+=1
print(f"SLOT1 bad_labels={badlab} label_gt8pct={big} rc_cases={rc1}")
print(f"vs 9771508: tumour_differs={tum_diff}/179(want0) rc_differs={rc_diff}/179(want>0) ref_rc={rc_ref}")
print("VERIFY_SLOTS_DONE")
