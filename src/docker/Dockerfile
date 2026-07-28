# BraTS-METS 2025/26 submission image — Dataset007 ResEnc-L (patch96/batch4, region-based)
# MUST be built for linux/amd64 (Synapse eval infra is amd64; we trained on aarch64 GB10).
#   docker buildx build --platform linux/amd64 -t docker.synapse.org/<PROJECT_ID>/brats-mets-resenc:v1 --load .
#
# Network is DISABLED at evaluation, so torch + nnunetv2 + the model weights are all baked in here.

# amd64 CUDA runtime base with torch preinstalled (satisfies nnunetv2 torch>=2.1.2,!=2.9.*)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# nnU-Net + IO/postproc deps (pinned to the version we trained with)
RUN pip install --no-cache-dir \
        nnunetv2==2.6.4 \
        SimpleITK==2.4.0 \
        nibabel \
        scipy \
        connected-components-3d

# nnU-Net expects these; only results is actually used at inference time
ENV nnUNet_raw=/opt/nnunet/raw \
    nnUNet_preprocessed=/opt/nnunet/preprocessed \
    nnUNet_results=/opt/nnunet/results \
    nnUNet_compile=0 \
    MODEL_DIR=/opt/nnunet/results/Dataset007_BraTSMets2025/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres

# Bake the trained model. The ./model dir must contain (copied from the training box):
#   plans.json  dataset.json  dataset_fingerprint.json
#   fold_all/checkpoint_final.pth
# (checkpoint_best/latest and debug.json are NOT needed — keep the image small.)
COPY model/ ${MODEL_DIR}/

COPY run_model.py /opt/run_model.py

ENTRYPOINT ["python", "/opt/run_model.py"]
CMD ["--input-dir", "/input", "--output-dir", "/output"]
