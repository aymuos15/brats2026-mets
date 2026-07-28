# BraTS-METS submission container — Dataset007 ResEnc-L

Packages our trained nnU-Net (ResEnc-L, patch 96³ / batch 4, region-based
WT/TC/ET/RC, fold `all`) as a Synapse model-to-data Docker image.

## 0. Assemble the model weights (run when training finishes)

The image bakes in the trained model from sie272. Copy ONLY what inference needs:

```bash
mkdir -p model/fold_all
M=~/projects/CAI4CAI/models/Dataset007_BraTSMets2025/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres
# from this laptop (amd64 build host); pull via the bouncer/fabric as available
rsync -a sie272-pc:$M/plans.json sie272-pc:$M/dataset.json sie272-pc:$M/dataset_fingerprint.json model/
rsync -a sie272-pc:$M/fold_all/checkpoint_final.pth model/fold_all/
```
Result layout the Dockerfile expects:
```
model/
  plans.json
  dataset.json
  dataset_fingerprint.json
  fold_all/checkpoint_final.pth
```

## 1. Build (MUST be amd64)

We trained on aarch64 (GB10); Synapse eval is amd64. Build host = this laptop (x86_64) or any amd64 box.

```bash
docker buildx build --platform linux/amd64 \
  -t docker.synapse.org/<PROJECT_ID>/brats-mets-resenc:v1 --load .
```

## 2. Test locally with prod-like isolation (no network)

```bash
docker run --rm --network none --gpus all \
  --volume $PWD/sample_data:/input:ro \
  --volume $PWD/output:/output:rw \
  --shm-size 2g \
  docker.synapse.org/<PROJECT_ID>/brats-mets-resenc:v1
```
`sample_data/` should hold a few cases in BraTS layout
(`<case>/<case>-t1c.nii.gz`, `-t1n`, `-t2f`, `-t2w`). Check `/output/<case>.nii.gz`
has integer labels {0,1,2,3,4}. (Drop `--gpus all` to verify CPU fallback works.)

## 3. Push to your Synapse project

```bash
cat ~/synapse.token | docker login docker.synapse.org --username <SYN_USER> --password-stdin
docker push docker.synapse.org/<PROJECT_ID>/brats-mets-resenc:v1
```
Requirements: Certified User, PAT with **Modify**, push to **your own** project.

## Knobs
- `MIN_COMPONENT_VOXELS` (env, default 5): conservative speck removal. The
  lesion-wise metric excludes <2 mm³ lesions and punishes false negatives, so
  keep this gentle — tune on a held-out BraTS set before trusting a larger value.
- TTA (mirroring) is ON. `tile_step_size=0.5`.

## ⚠️ Confirm against the official BraTS-METS spec before final submission
- Exact `/input` layout (per-case subfolder vs flat) and modality filename pattern.
- Exact `/output` filename (`<case>.nii.gz` vs `<case>-seg.nii.gz`).
- Memory/GPU limits and time budget of the eval harness.
Both are easy to adjust: see the CASE-DISCOVERY and OUTPUT-NAMING blocks in `run_model.py`.
