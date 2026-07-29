# Handoff — BraTS 2026 Task 1 (Brain Metastases)

_Written 2026-07-28. Supersedes nothing in `docs/BraTSHandoff2026.md`, which remains the record of
the modelling work up to 2026-07-24. This file covers the submission artifacts._

---

## 0. The only thing that matters

**Deadline: 2026-07-30 23:59 UTC.** Three artifacts, all due together:

| # | artifact | state |
|---|---|---|
| 1 | **Short paper** (OpenReview) | draft complete, compiles, 3 items outstanding — §2 |
| 2 | **Docker container** (docker.synapse.org) | **PUSHED + SUBMITTED 07-29** as `9774087` — §3 |
| 3 | **Signed copyright form** | organisers share it; timing is ambiguous — §5 |

**No paper ⇒ the Docker is never run ⇒ no final ranking.** The two are linked by our Synapse team
name, **3495063**, which must be given on the submission form.

The scoring work is finished and stable. Every decode and ensemble axis was closed by 07-24
(`docs/BraTSHandoff2026.md`). Do not reopen them; spend the remaining time on 1–3 above.

---

## 1. What this repository is

Consolidated 2026-07-28 from three places that had drifted apart: `sie271:~/brats2025/` (the real
pipeline), the laptop (`brats_hint/`, `brats_mets_submission/`), and the custom trainers living
inside the nnU-Net site-packages tree on sie271.

```
src/decode/       3-way fusion, hysteresis, RC graft, submission assembly   <- the method
src/train/        custom trainers + dataset builders (D007/011/014/015)
src/eval/         local harness vs the official metric, submit + verify
src/diagnostics/  FP characterisation, miss diagnosis, extent measurement
src/experiments/  levers that did not work, kept for the paper's evidence
src/docker/       amd64 container reproducing the submitted decode
src/setup/        environment + data preparation
paper/            LNCS short paper
results/          our own decode sweeps and figures
docs/             the modelling handoff up to 07-24
```

**Git history was rebuilt as a single commit on 07-28** and there is **no remote**. See §4 before
adding one.

---

## 2. The paper (`paper/`)

`paper/main.tex`. Build with `latexmk -pdf -outdir=build main.tex`. Needs `llncs.cls`, which is
**not** in the repo (Springer's licence forbids redistribution) — fetch from `bit.ly/2TEcZNF` or
copy from another project.

**Current state:** compiles clean, no overfull boxes, no undefined references. **Content ends on
page 10, references begin on page 11.** The limit is **8–10 pages excluding references**, verified
against the wiki via the Synapse API (the web view is a JS shell and fetches empty):

> "Your paper must be 8 - 10 page long (without references)" … "Citations do not count toward the
> paper length limit."

Acknowledgements *do* count. There is no page budget left — anything added must displace something.

**Outstanding, in priority order:**

1. **Training-set results table.** The rules require results on training *and* validation data.
   Validation is Table 2; there is no training table. This is the one **unmet hard requirement**.
   Removing the Limitations section freed roughly the half page it needs.
2. **Public GitHub repo.** The paper cites `github.com/aymuos15/brats2026-mets`. It does not exist
   yet, so the required source-code link does not resolve. See §4.
3. **BraTS 2026 challenge manuscript citation** — organisers release it and require it be cited.
   Marked as a TODO in the bibliography; re-check wiki page 639585 before camera-ready.

**Framing decisions made deliberately — do not reverse without reading §4:**

- **No challenge standings anywhere.** Not our placement, not per-metric ranks, not mean rank, no
  comparison against the field. The rules permit publishing our own results but **not** overall
  challenge results or analysis of them until the organisers' overview paper appears. The paper
  argues from our own metric values throughout.
- **Framed as contributions, not negative results.** The confidence-inversion finding is stated as
  a *design rule*; the harness finding as a *validation protocol*. Evidence and numbers are
  unchanged — only the register.
- **Limitations section removed** on 07-28 by decision. Three caveats survive inline (fold_all
  training, threefold optimism of offline forecasts, absence of per-case validation scores). Two
  are now unstated: the shared lineage of the ensemble members, and that findings are specific to
  lesion-wise scoring.

**Not double-blind.** No anonymity language anywhere in the wiki, and the rules *require* a
source-code link in the paper, which cannot coexist with anonymisation. Author names are in.

---

## 3. The Docker container (`src/docker/`)

`Dockerfile.2026` + `run_inference.py`. Reproduces submission `9771992`: three sequential nnU-Net
passes → 3-way probability average → hysteresis `lo 0.25 / hi 0.70` → RC from the CC-loss member,
keep-largest.

### Measured, 2026-07-28, on sie236 (RTX 4090, pinned to `device=0`)

```
member topk:       7.3 s/case
member completion: 7.0 s/case
member ccloss:     7.0 s/case
fusion:            0.1 min
[verify] outputs=6/6 bad_labels=0 rc_dominating=0 output_flat=True
TOTAL 2.2 min (22.4 s/case)          VERIFY_GATE: PASS
```

**22.4 s/case → roughly 67 min for 179 cases on a 4090.** The challenge grants **12 h total** on an
**A10G**, which is materially slower; even at 5× slower this lands near 5.5 h. Comfortable.

⚠️ **Caveats on that number:** it was **6 cases, not 179**, and the A10G scaling is an *estimate*,
not a measurement. If you want certainty, run the full 179 — the input set is at
`sie271:~/brats2025/extracted/Validation/` in exactly the `/input` layout the organisers use.

### Two traps this cost time to find

1. **Must build `--platform linux/amd64`.** We train on aarch64 GB10s; the eval infra is amd64.
   sie236 is the x86_64 box with Docker.
2. **Stock pip `nnunetv2` lacks our trainer classes**, and nnU-Net resolves the trainer *by name
   from the checkpoint* at inference. The image overlays our own `nnunetv2` tree
   (`nnunetv2_custom/`) to fix this. Without it the container dies immediately with
   `Unable to locate trainer class nnUNetTrainerDiceTopK10BCE`.

### State on sie236

```
~/brats2026/build/    build context (2.3 GB: 3 checkpoints + nnunetv2_custom)
~/brats2026/input/    6 validation cases, BraTS /input layout
~/brats2026/output/   6 predictions from the timing run
~/brats2026/run.log   the timing output above
docker image:         brats-mets-3way:v1  (9.46 GB)
```

The build context is **not** in this repo (2.3 GB of weights, gitignored at `.docker-build/` on the
laptop). Checkpoints came from `sie271:~/projects/CAI4CAI/models/`:

| member | dataset / trainer | checkpoint |
|---|---|---|
| topk | D007 `nnUNetTrainerDiceTopK10BCE` | `checkpoint_topk_snap.pth` |
| completion | D011 `nnUNetTrainerDiceTopK10BCE` | `checkpoint_snap_ep946.pth` |
| ccloss | D011 `nnUNetTrainerCCLoss_BackSplit_FT` | `checkpoint_best.pth` |

Each is renamed to `fold_all/checkpoint_final.pth` inside the image so the entrypoint can use one
name.

### ✅ Pushed and submitted 2026-07-29

```
image      docker.synapse.org/syn75900400/brats-mets-3way:v1
digest     sha256:223d6de4bd00cf4b4734bdd3d5900c84767641ba20667a2b3dea2f43124852d0
repo entity syn76483331   (project syn75900400)
queue      9619627  "Task 1: Brain Metastases - Docker"
submission 9774087  "3way_lo025hi070_ccRC"   status RECEIVED
```

Decode baked in is `LO 0.25 / HI 0.70`, 3-way fusion, RC from CC-loss keep-largest — i.e. our best
validation submission `9771992`. **The container is only run if the short paper is submitted with
Synapse team name 3495063 on the OpenReview form.**

Test with the organisers' exact command; `--gpus '"device=0"'` keeps the 5090 free on sie236:

```bash
docker run --rm --network none --gpus '"device=0"' \
  --volume ~/brats2026/input:/input:ro --volume ~/brats2026/output:/output:rw \
  --memory=48G --shm-size=16G brats-mets-3way:v1
```

⚠️ **Organisers do not test containers until after the queues close.** A container that fails is
simply not evaluated, with no error message and no second chance.

---

## 4. Publication constraints — read before pushing anything

**The challenge embargoes overall challenge results.** We may publish our own method's results and
our own team ranking; we may **not** publish overall challenge results or analysis of them until
the organisers' overview paper is out.

This has a concrete consequence. `lb_snapshot.csv`, `lb.json`, `lb_1400.csv` and
`brats2026final.html` contain **every submission by every team** (2,076 submissions, 143
submitters at the last pull). They are:

- **gitignored** and **absent from git history** (history was rebuilt as a single commit on 07-28
  specifically to remove them),
- kept locally at **`../.brats-private-leaderboard/`** on the laptop,
- regenerable on sie271 with `lbquery2.py` (see §6).

**Before making the repo public**, re-verify:

```bash
git log --all --name-only --format="" | sort -u | grep -E "lb_snapshot|lb\.json|lb_1400|brats2026final"
git grep -l "eyJ0e""XAi" $(git rev-list --all)   # Synapse token (JWT prefix)
```

The first must return nothing. The second is written with a broken string literal so that this
file does not match its own check; if you retype it whole, expect `HANDOFF.md` itself as a hit and
treat **any other file** as a real one. The Synapse token is **not** in the repo; the copy of the handoff at
`docs/BraTSHandoff2026.md` has it redacted (the original on the laptop still contains it).

`../.brats-staging/` holds the rsync staging copy from sie271 and also contains the dumps. It was
moved out of the repo directory for this reason.

---

## 5. Timeline and obligations

| date | what |
|---|---|
| **Jul 30, 23:59 UTC** | **Docker + short paper + copyright form.** All queues close, including validation |
| Aug 6 | Reviews available, initial decision |
| Aug 13 | Revised-paper deadline (Revision decisions only) |
| Aug 24 | Final acceptance |

We are **Track B (Standard)**. Track A (Early) closed Jul 2 and using it would have *barred* us
from submitting on Jul 30.

- **Copyright form timing is inconsistent in the documentation** — the Instructions page says "at
  camera-ready", the Timeline says "at final submission". Unresolved; ask on the Community tab.
- **Peer review duty:** after submissions close we are invited to lightly review 2–3 papers.
- **Camera-ready** is the identical paper plus test results. Nothing in the current draft depends
  on the test set.
- **Withdrawing the paper** also drops us as co-authors on the BraTS journal manuscript.
- Organisers may publish our container under **Apache 2.0** unless we state otherwise.

---

## 6. Machines and commands

| host | role | notes |
|---|---|---|
| `sie209-lap` (laptop) | orchestration, this repo | x86_64, Docker + buildx |
| `sie271-pc` | pipeline, data, Synapse tooling | GB10 @ **611 MHz**, ~4× slower, not fixable |
| `sie272-pc` | fast GPU | GB10 @ 2400 MHz. Runs someone else's thesis — **do not kill** |
| `sie236` | **the amd64 Docker box** | x86_64, RTX 4090 (idx 0) + 5090 (idx 1). Log in as `tosin` |

⚠️ `sie271:~/brats2025/.venv` is **hidden** — a `~/brats2025/*` glob misses it. This is how the
Synapse tooling once got written off as lost.

```bash
# leaderboard (private use only, see §4)
cd ~/brats2025
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring ./.venv/bin/python lbquery2.py
./.venv/bin/python rank12.py      # 12-metric standing, both conventions
./.venv/bin/python ours.py        # our scored submissions by date

# read the challenge wiki (the web page is a JS shell and fetches empty)
#   syn74274097 pages: 639582 Submission Instructions, 639585 Rules,
#                      639587 Timeline, 640600 Comprehensive Timeline
#   via syn.restGET(f"/entity/syn74274097/wiki/{page_id}")["markdown"]

# submit a validation zip (needs the Synapse .venv, not the numpy decode env)
./.venv/bin/python submit.py <zip> <name>
```

---

## 7. Corrections to earlier documents

- **`docs/BraTSHandoff2026.md` quotes the wrong NSD row.** It gives `.739/.811/.798/.521`, which
  belongs to submission `9771508`. The correct row for our best entry `9771992` is
  **`.741/.815/.806/.521`** (wt/tc/et/rc), pulled from the scored table on 07-28. The paper uses
  the correct values.
- **`erode_cost.py` and `score_rc_native.py` do not exist** on sie271, though the handoff
  references both. The erosion `k=1` finding is documented but not reproducible from code.
- **`config_mets_nohd.yaml`** (the panoptica-hang workaround) is also absent; only the stock
  configs were recovered.

---

## 8. Open decisions

1. **Training-set results table** — the unmet hard requirement (§2). We chose *not* to compute
   per-case statistics: validation per-case scores are withheld by the server, and our own 80-case
   harness runs on `fold_all` models that memorised those cases, so any mean ± std would describe
   performance on memorised data. Paired *decode* comparisons on identical probability maps would
   still be valid, since memorisation is common-mode there.
2. **Restore one sentence on shared ensemble lineage?** Removed with the Limitations section, but
   it is the reason the ensemble cannot dilute these particular false positives. Would sit better
   in §2.3 (dilution) than as a limitation.
3. **Full 179-case container timing** — currently extrapolated from 6 cases (§3).
