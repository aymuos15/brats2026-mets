import glob, os, numpy as np, nibabel as nib
SP="/home/ssk23/brats2025/diffchk"
LBL={1:"NETC",2:"SNFH",3:"ET",4:"RC"}
tot_old={k:0 for k in LBL}; tot_new={k:0 for k in LBL}; changed=0; ncase=0
for f in sorted(glob.glob(SP+"/old/*.nii.gz")):
    b=os.path.basename(f)
    a=np.asarray(nib.load(f).dataobj)
    c=np.asarray(nib.load(SP+"/newA/"+b).dataobj)
    for k in LBL: tot_old[k]+=int((a==k).sum()); tot_new[k]+=int((c==k).sum())
    changed+=int((a!=c).sum()); ncase+=1
print(f"cases: {ncase}   voxels differing between 9771150 and candidate A: {changed:,}")
print(f"{'label':>6} {'9771150':>12} {'A lo0.30':>12} {'delta':>12} {'pct':>7}")
for k in LBL:
    d=tot_new[k]-tot_old[k]
    p=100.0*d/max(tot_old[k],1)
    print(f"{LBL[k]:>6} {tot_old[k]:>12,} {tot_new[k]:>12,} {d:>+12,} {p:>+6.1f}%")
print("\nRC identical (untouched by tumour hysteresis):", tot_old[4]==tot_new[4])
