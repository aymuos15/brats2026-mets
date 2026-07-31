# BraTS 2026 paper review

## Overall assessment

This is already a strong challenge paper. It has a clear central result, a coherent inference-time story, useful negative findings, and enough quantitative analysis to justify publication in a challenge proceedings. It does not need a new experiment before submission. The priority is to reduce claims that exceed the evidence and to fill a few reproducibility gaps.

## Must fix before submission

### 1. Verify the organiser-mandated citations and exact data release

The source still contained a TODO about the required BraTS 2026 citation. I added provisional citations for the BraTS-METS 2025 task manuscript and the BraTS 2026 cluster description, but the corresponding author should check the current Challenge Rules immediately before submission.

Also state whether training used:

- the separate corrected-label overlay;
- the later QCed UCSD training-set release;
- any handling of the out-of-range label reported in the challenge discussion.

A single sentence in the Data subsection is enough.

### 2. Narrow the confidence-based “design rule”

Original claims such as “any criterion monotone in confidence is ruled out for this task” are much broader than the evidence. Seven methods on one system show that the tested size/confidence heuristics failed for these models. They do not rule out every confidence-aware method or establish a task-wide law.

The comparison of false-positive peak probability with the peak inside missed reference lesions is also conditioned on the decoding threshold. It is useful evidence that simply raising the threshold will not solve both errors, but it is not a general demonstration that confidence is inverted with correctness.

### 3. Replace the universal validation protocol with an empirical warning

The original distinction between “thresholds/fusion weights change values” and “component rules change counts” is not technically valid. Thresholding and fusion can also create, remove, split, or merge connected components.

The defensible result is narrower:

- the training-case harness was not an unbiased validation set;
- rankings transferred reasonably for the tested threshold/fusion settings;
- one count-expanding cavity rule reversed sign on held-out validation;
- therefore all such tuning is provisional, with particular caution for component-retention rules.

The reviewed version uses this framing.

### 4. Do not conclude that false positives *are* anatomical mimics

The paper has non-expert review on single axial slices and a weak surface-distance discriminator (AUC 0.575). This supports “consistent with anatomical mimics” or “suspected anatomical mimics,” not “we have shown they are anatomical mimics.” The original conclusion overstates the evidence.

### 5. Add minimal method details for the ensemble members

The current Methods section does not explain enough for a reader to reconstruct the models. Add brief definitions of:

- what the “6-region” configuration predicts;
- what “completion” means;
- the value/fraction used for top-K BCE;
- the connected-component loss formulation or exact cited implementation;
- the fine-tuning duration/checkpoint selection;
- how the six outputs are mapped to WT/TC/ET/RC.

This can be two or three sentences. It is the largest reproducibility hole.

### 6. Clarify the development table

The sequence currently says row 2 is “+ ensembling,” row 3 adds a “second member,” and row 6 adds a “third member.” A reader cannot determine what row 2 actually ensembles. Rename the rows so each configuration is explicit.

Also describe this as a sequential development trace rather than a factorial ablation. The row-to-row effects are conditional on previous changes.

### 7. Correct the fusion-weight table label

The 1/3 row has 0.720/0.761/0.737, whereas the final submitted configuration after lowering the extent threshold has 0.725/0.766/0.746. Calling the 1/3 row “submitted” is therefore misleading. The reviewed version labels it “before extent-threshold refinement.”

### 8. Avoid implying a fixed official aggregate over 12 metrics

The paper says the selected point was best on “the aggregate over all twelve official metrics.” Unless the authors explicitly computed an unweighted mean for internal selection, call it the “best overall balance across the 12 reported cells.” Do not imply this reproduces the official final ranking procedure.

### 9. Define the error-analysis denominators

For the 80-case analysis, specify:

- which tumour regions are pooled;
- the official definition of “small”;
- the distance threshold behind “86% fall far from any reference lesion”;
- whether probabilities are measured before or after hysteresis;
- how false-positive and missed components are identified.

Without these definitions, the most interesting statistics are not reproducible.

### 10. Give exact inference hardware

Replace “consumer GPU” with the GPU model and state whether 22 seconds includes preprocessing, sliding-window inference, post-processing, and disk I/O.

## Should fix

- The abstract was 294 words and very dense. The reviewed version is shorter and less promotional.
- “No new training run” was misleading because a three-model ensemble necessarily depends on multiple trained models. “Without retraining the constituent models” is safer.
- Remove the unsupported 0.005 “resolution” threshold for interpreting Dice differences. Deterministic aggregate scores have no measurement noise; the limitation is absence of per-case uncertainty.
- Remove the unexplained boldface on RC small-instance F1 = 0.167.
- Use leading zeros consistently in tables.
- Use `t_{\mathrm{lo}}` and `t_{\mathrm{hi}}`; plain `$lo$` and `$hi$` are typeset as products of variables.
- “Strict trade-off” should be “monotone over the tested weights.” Three sampled points do not establish a strict general relationship.
- “Every newly detected small lesion” should be “many newly detected small lesions” unless this was literally verified for every added true positive.
- “Under volumetric Dice the graft would be nearly free” is an untested counterfactual. “Would receive much less weight” is sufficient.
- Consider deleting the paragraph listing MedNeXt-v2, KiU-Net, U-Mamba, larger kernels, and other abandoned ideas unless the underlying results are available. It is interesting, but currently anecdotal.

## Formatting and compile check

- The original source compiles successfully.
- It produces 11 PDF pages total.
- References begin on page 11, leaving 10 pages before references.
- The reviewed source also compiles to 11 pages total, with references beginning on page 11.
- No missing references or fatal LaTeX errors remain after two compilation passes.
- Two very small overfull boxes remain (about 2.3 pt in prose and 1.4 pt in the ensemble table); neither is visibly problematic.

## Suggested final triage

1. Fill the five `% REVIEW TODO:` comments in `main_reviewed.tex`.
2. Clarify rows 2–3 of the development table.
3. Verify mandatory citations and the exact data-release wording.
4. Confirm every reported delta and whether single values denote WT or a mean across WT/TC/ET.
5. Submit. Do not invent another experiment tonight.
