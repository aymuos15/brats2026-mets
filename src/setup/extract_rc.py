"""Extract ONLY the 167 cavity-bearing training cases from the 33 GB zip.

The local eval cohort has zero RC, which makes the harness blind to 2 of the 11
ranked metrics (RC DSC + RC NSD). We sit at .630; teams #3 and #4 post .671.
The cases were never missing -- they were in an unextracted site folder inside
the zip the whole time.
"""
import json, os, zipfile, sys

HOME = os.path.expanduser("~")
ZIP = f"{HOME}/brats2025/raw_zips/MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip"
OUT = f"{HOME}/brats2025/extracted_rc"
ids = [i.replace("sub-", "BraTS-MET-") for i in json.load(open(f"{HOME}/brats2025/rc_cases.json"))]
want = set(ids)
os.makedirs(OUT, exist_ok=True)

print(f"[rc] {len(want)} cavity cases wanted", flush=True)
zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
print(f"[rc] zip holds {len(names)} entries", flush=True)

members, found = [], set()
for n in names:
    if n.endswith("/"):
        continue
    base = os.path.basename(n)
    for cid in want:
        if base.startswith(cid + "-") and base.endswith(".nii.gz"):
            members.append((n, cid, base))
            found.add(cid)
            break

print(f"[rc] matched {len(members)} files across {len(found)} cases", flush=True)
missing = want - found
if missing:
    print(f"[rc] WARNING {len(missing)} cases not in zip: {sorted(missing)[:5]}")

for i, (n, cid, base) in enumerate(members):
    d = f"{OUT}/{cid}"
    os.makedirs(d, exist_ok=True)
    dst = f"{d}/{base}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        continue
    with zf.open(n) as src, open(dst, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if (i + 1) % 100 == 0:
        print(f"  ..{i+1}/{len(members)}", flush=True)

ncase = len([d for d in os.listdir(OUT) if d.startswith("BraTS-MET-")])
nseg = sum(1 for c in os.listdir(OUT) if os.path.exists(f"{OUT}/{c}/{c}-seg.nii.gz"))
print(f"[rc] DONE  cases={ncase}  with-seg={nseg}  -> {OUT}")
print("EXTRACT_RC_COMPLETE")
