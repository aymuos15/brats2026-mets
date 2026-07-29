# Short paper — BraTS 2026 Task 1

**Deadline: 2026-07-30 23:59 UTC.** Mandatory. Per the Submission Instructions wiki:
*"Failure to provide a short paper will disqualify your Docker submission from evaluation and
final ranking"* and *"We will only run Docker submissions linked to a short paper."*

Submit to **OpenReview** → <https://openreview.net/group?id=MICCAI.org/2026/Challenge>

## Building

`main.tex` requires the Springer LNCS class, which is **not** bundled here (Springer's licence
does not permit redistribution). Download `llncs.cls` from <https://bit.ly/2TEcZNF> and place it
next to `main.tex`, then:

```bash
pdflatex main && pdflatex main
```

## Hard requirements checklist

- [ ] **8–10 pages excluding references.** Citations do not count toward the limit.
- [ ] Springer LNCS template.
- [ ] Section order: Abstract · Keywords · Introduction · Methods · Results · Discussion ·
      Acknowledgements (optional) · References.
- [ ] **No citations in the abstract** — this is an explicit rule, and `main.tex` carries a
      comment marking it.
- [ ] **Synapse team name `3495063` stated on the OpenReview submission form.** If this is wrong
      the container is treated as paperless and never run.
- [ ] Source-code GitHub link in the paper. Currently `github.com/aymuos15/brats2026-mets` —
      **the repository is not public yet**, and the link must resolve before submission.
- [ ] Results on **training AND validation** data.
- [ ] Acknowledgement string verbatim: *"Data used in this publication were obtained as part of
      the Challenge project through Synapse ID (syn74274097)."*
- [ ] Mandatory citations present: BraTS 2021 flagship (arXiv:2107.02314), BraTS-MET 2023
      (arXiv:2306.00838), MedPerf (Nat Mach Intell 5:799–810, 2023).

## Open TODOs in the draft

1. **ORCIDs** — not yet in `main.tex`. Author names, affiliations and the corresponding
   e-mail (`soumya_snigdha.kundu@kcl.ac.uk`) are confirmed.
2. **NSD row in Table `tab:final` is `TODO`.** The handoff records NSD `.739/.811/.798/.521` for
   submission `9771508`, *not* for our best `9771992`. Pull the correct row before submitting:
   ```bash
   ssh sie271-pc 'cd ~/brats2025 && ./.venv/bin/python ours.py'
   ```
3. **Training-set results.** The challenge requires results on training data as well as
   validation. The draft currently reports the local-harness setup but not a training results
   table. Either add one from the harness (quoting the §2/§5 limits, which the Discussion already
   explains) or state explicitly that training-set numbers are from memorised `fold_all` models.
4. **BraTS 2026 challenge manuscript citation** — organisers release it and require it to be
   cited. Marked as a TODO comment in the bibliography; re-check the Challenge Rules wiki page
   before camera-ready.
5. **Page count not yet checked** — cannot compile here without `llncs.cls`.

## After submission

- **Peer review duty:** we will be invited to lightly review 2–3 papers.
- **Camera-ready** = this identical paper plus the test-set results. Nothing in the current
  draft depends on the test set.
- **Top-ranked methods must have a published article** (a preprint counts) before the results
  announcement; this short paper can serve as it.
- **Embargo:** we may publish our own results and our own team ranking, but not overall challenge
  results or analysis until the organisers' overview paper is out.
- **Withdrawal** means being dropped as co-authors on the BraTS journal manuscript.
