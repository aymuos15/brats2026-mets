# BraTS 2026 — Task 1 (Brain Metastases) — CAI4CAI

Source code for our submission to the **BraTS 2026 Challenge, Task 1 (Brain Metastases
Segmentation)**. Synapse team **3495063**.

Our final entry is an **inference-time method**: three independently-trained nnU-Net ResEnc-L
variants are probability-averaged and decoded with **hysteresis thresholding**, with the
resection-cavity channel grafted from a connected-component-loss model. Neither of the two
largest gains in our development history required a new network.

---

## Method

### Networks (3-way fusion)

All three members are `nnUNetResEncUNetLPlans` / `3d_fullres`, trained on
`Dataset007_BraTSMets2025` — a 4-region formulation with
`WT=[1,2,3]`, `TC=[1,3]`, `ET=[3]`, `RC=[4]`, `regions_class_order=[2,1,3,4]`.

| member | trainer | source |
|---|---|---|
| TopK 4-region | `nnUNetTrainerDiceTopK10BCE_FT150` | `src/train/` |
| completion | `nnUNetTrainerCCLoss_BackSplit_FT` (D011) | `src/train/` |
| CC-loss FT | `nnUNetTrainerCCLoss_BackSplit_FT` (D011) | `src/train/` |

Inference: `step_size 0.25`, TTA on. Decode order WT(2) → TC(1) → ET(3) → RC(4).

### Decoding — hysteresis (the single largest lever)

```python
def hyst(p, lo, hi):                       # lo = 0.25, hi = 0.70
    lab, n = cc_label(p > lo)              # components of the GENEROUS mask
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())   # PEAK probability inside each
    keep = cmax > hi                       # keep it iff that peak clears HI
    keep[0] = False
    return keep[lab]                       # paint the WHOLE component back
```

`hi` decides **whether** a lesion exists (one confident voxel suffices); `lo` decides **how far
it extends**. No single threshold does both. Critically this filters on **peak** confidence and
never on size — unlike the size/mean-confidence component filters that failed four times for us
(see *Negative results*).

Applied per region; **RC excluded**. RC is thresholded at 0.50 and taken from the CC-loss member
(keep-largest).

Implementation: `src/decode/assemble_3way.py`, `src/decode/ensemble_3way.py`.

---

## Results (BraTS 2026 official validation set, 179 cases)

Best submission **`9771992`** — 3-way fusion, hysteresis `lo 0.25 / hi 0.70`.

| | WT | TC | ET | RC |
|---|---|---|---|---|
| **Lesion-wise DSC** | .725 | .766 | .746 | .630 |
| **Lesion-wise NSD** | .741 | .815 | .806 | .521 |
| **Small-instance F1** | .376 | .478 | .476 | **.167** |

Overlap and surface metrics are strong; the three tumour F1 cells are the residual gap, and
detection rather than delineation is the open problem.

> Note: challenge standings are not reported here. Per the challenge rules, participants may
> publish their own method's results but not overall challenge results or analysis of them until
> the organisers' overview paper is released.

### Ablation path

`baseline nnU-Net ResEnc-L → RC-fix → TopK-BCE → overlap .375 → ENSEMBLING → recall-repaired
member → overlap .25 → completion member → RC graft → HYSTERESIS → 3-WAY FUSION`

Decode sweeps over our own submissions: `results/ablations/decode_scores*.json`.

### The DSC ↔ F1 frontier

Interpolating the CC-loss member's fusion weight traces a strict frontier — F1 can be bought only
by paying DSC, and the optimum is a single point:

| CC-loss weight | DSC wt/tc/et | F1 wt/tc/et |
|---|---|---|
| ⅓ (submitted) | **.720/.761/.737** | .376/.479/.477 |
| 0.42 | .695/.738/.714 | .403/.504/.510 |
| ½ | .670/.715/.691 | **.429/.530/.543** |

---

## Negative results

We consider these the most transferable part of this work. On 80 ground-truth cases scored with
the official metric:

| | |
|---|---|
| median peak probability inside our **false positives** | **0.96** |
| median peak probability inside lesions we **miss** | 0.55 |
| misses the model is blind to (peak < 0.10) | 43% |
| FPs nowhere near any GT lesion | 86% |

**The model is more confident about its mistakes than about the lesions it finds.** Our false
positives are confident, spatially coherent (they survive Gaussian smoothing), isolated, and
*corroborated* — all three independently-trained members fire on them.

Seven FP-suppression mechanisms were refuted, all for the same reason — each separates on
confidence, or on something monotone in it:

| mechanism | outcome | code |
|---|---|---|
| small-component filters | annihilated F1; small **is** real | `src/experiments/` |
| FP-rejection by mean confidence | recovered DSC, killed the recall it was meant to keep | `src/experiments/score_v2.py` |
| CC-loss v2 (per-component FP penalty) | FP −53% but TP −24% — collapsed into global under-confidence | `src/train/nnUNetTrainerCCLossV2_FT.py` |
| v2 as a learned veto | rejected everything at t=0.5 (TP 1.82 → 0.51) | " |
| v1/v2 blend | FPs and TPs removed in lockstep; no sweet spot | `src/experiments/score_blend.py` |
| size-adaptive seed | reproduces the plain global threshold exactly | `src/decode/hyst_sweep_probe.py` |
| scale-space (smoothed) seed | plain threshold detects more at every matched FP rate | " |

Two further negative results:

- **Small-lesion oversampling (`Dataset014_BraTSMetsSmallCC`)** — patches centred on a small
  lesion went 1.28% → 25.75% (20×, verified over 4,000 draws). The blindness did not move.
  The network is not blind to small metastases because it rarely sees them.
  Code: `src/train/build_d014.py`, `src/train/nnUNetTrainerSmallCC_FT.py`.
- **Target dilation (`Dataset015_BraTSMetsDilateISO`)** — enlarging isolated small targets *did*
  lift tumour F1 (+.012/+.014/+.013 standalone), but grafting those detections into the fusion
  cost more lesion-wise DSC than the F1 was worth. Every grafted small lesion enters the
  per-lesion DSC average below the mean. Code: `src/train/build_d015.py`,
  `src/decode/assemble_d015_union.py`.

### FP characterisation

`src/diagnostics/fp_crops.py` (montage → `results/figures/fp_montage.png`) and
`src/diagnostics/fp_surface_dist.py`. The FPs are predominantly small bright enhancing foci at
the cortical margin/dura, on vessels, and at the midline — i.e. **mimics**, not annotation gaps.
But they are not *anatomically separable* either: FP centroid distance-to-brain-surface (median
8.8 mm) versus GT lesions (11.2 mm) gives **AUC ≈ 0.575**, far short of what a usable filter needs.

---

## Local evaluation harness — and its limits

`src/eval/` wraps the official `BraTS_evaluation` + panoptica with `config_mets.yaml`
(`src/eval/brats_eval_configs/`). Two sets: 80 GT tumour cases, 100 GT cavity cases.
**Validation gate:** the harness must reproduce the relative ordering of our own scored
submissions; if a change breaks that, the harness is wrong.

Three limits, learned expensively:

1. **Everything is `fold_all` — there is no held-out split.** The models were trained on these
   cases. The harness is valid for comparing **decodes** (same probabilities, different
   post-processing — memorisation is common-mode and cancels when candidates are ordered), and **not** for
   comparing models or reading absolute scores.
2. **Direction yes, magnitude no.** It consistently overestimated the size of the gains it
   predicted, by roughly a factor of three.
3. 🔴 **Never tune anything that changes how many components you keep.** On memorised cases the
   model rarely emits a spurious component, so a "keep more components" rule looks free — TP up,
   FP flat. On unseen data it emits them and the rule admits them. *Memorisation suppresses
   exactly the false positives such a rule would let in, so the harness is structurally blind to
   the cost.* This produced a **wrong-sign** result for us: a cavity keep-rule measured at
   RC-DSC +0.028 locally and delivered **−0.016** on the validation set. Fusion weights and
   thresholds are safe (they change values, not counts); component filters, keep-rules and size
   cutoffs are not.

---

## Repository layout

```
src/decode/        3-way fusion, hysteresis, RC graft, submission assembly   <- the method
src/train/         custom nnU-Net trainers + dataset builders (D007/011/014/015)
src/eval/          local harness, official-metric scoring, submission + verify
src/diagnostics/   FP characterisation, miss diagnosis, extent measurement
src/experiments/   refuted levers, kept for the negative results above
src/docker/        container entrypoint (2025 baseline — being rebuilt for 2026)
src/setup/         environment + data preparation
results/           our own scored submissions, decode sweeps, figures
docs/              development handoff (full chronological record)
```

### Reproducing

```bash
source src/setup/nnunet_env.sh     # nnUNet_raw / _preprocessed / _results; export nnUNet_compile=0
bash  src/train/predict_members.sh # three members -> cached probabilities
python src/decode/assemble_3way.py # fusion + hysteresis lo .25 / hi .70 + RC graft -> submission
python src/eval/verify_slots.py    # 179 masks, labels in {0..4}, no label >8% of volume
```

Requires nnU-Net **v2.6.4**. Leaderboard tooling additionally needs `synapseclient` and a Synapse
personal access token — **read from `~/brats2025/syn.tok`, never committed to this repository.**

---

## Data

BraTS-MET 2025/2026 training set, 1295 cases. Region prevalence: NETC 54%, SNFH 89%, ET 96%,
**RC 13%**.

> Note for anyone reproducing: the training archive holds 1,296 cases across **two site
> folders**. Every resection-cavity case lives in the second (`UCSD - Training`). Unpacking only
> the first yields 651 cases and no cavities at all.

---

## Citation

If you use this code, please cite the BraTS benchmark and BraTS-MET:

- U. Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and
  Radiogenomic Classification*, arXiv:2107.02314.
- *BraTS-METS 2023 Challenge*, arXiv:2306.00838.
- A. Karargyris, R. Umeton, M. J. Sheller, et al., *Federated benchmarking of medical artificial
  intelligence with MedPerf*, Nature Machine Intelligence 5:799–810 (2023).

## Acknowledgement

> Data used in this publication were obtained as part of the Challenge project through Synapse ID
> (syn74274097).
