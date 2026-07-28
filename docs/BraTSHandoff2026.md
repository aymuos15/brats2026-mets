# BraTS 2026 — Task 1 (Brain Metastases) — Handoff

_Last updated: **2026-07-24 09:00**. Read §0 and §1 before touching anything. Everything before 07-13 in the old handoff is superseded._

> # 🚨 07-24 delta — THE DEADLINE MOVED TO **JUL 30**, AND §0 IS WRONG ABOUT THE PAPER. READ THIS FIRST.
>
> **Two facts invalidate the endgame plan this document was written around. Both verified 07-24 against the
> challenge wiki (`syn74274097`, pages 639582 / 639587 / 640600 / 639585) — the authoritative source, not the
> web portal (`challenges.synapse.org/brats2026` is a JS SPA and fetches empty).**
>
> ### 1. 🔴 THE ROUND DID NOT CLOSE ON 07-23 — the final deadline is **2026-07-30 23:59 UTC**
> The Timeline wiki shows `-- Jul 23 --` struck through in red, replaced by **Jul 30**. Every queue —
> *including the Validation Leaderboard* — closes then. Confirmed empirically: the board took **52 subs on
> 07-23 and 11 on 07-24**. **§0's "Round closes 2026-07-23" is superseded. We have ~12 more validation slots.**
> We are on **Track B (Standard)**; we never used the Jul 2 Early track, so Jul 30 is our one and only gate.
>
> ### 2. ☠️ §0's "Paper + Docker are due in AUGUST — NOT a July gate, not a DQ risk" IS FALSE. IT IS *THE* DQ RISK.
> All three artifacts are due **Jul 30**, together. The August dates (Aug 6 reviews, Aug 13 revisions,
> Aug 24 acceptance) are **post-submission review milestones**, not the submission gate. Verbatim from the
> Submission Instructions wiki:
> > "❗️ Submission of this short paper is **mandatory**. Failure to provide a short paper will **disqualify
> > your Docker submission** from evaluation and final ranking."
>
> And: *"We will only run Docker submissions linked to a short paper."* **No paper → the container is never
> run → we have no final ranking at all, whatever the validation board says.** Neither artifact appears to
> exist yet, and this document actively told the next agent not to work on them. **That instruction is rescinded.**
>
> | # | due Jul 30 23:59 UTC | where |
> |---|---|---|
> | 1 | Containerized algorithm (Docker) | `docker.synapse.org/PROJECT_ID/IMAGE:TAG` → Task 1 Submission tab |
> | 2 | Short paper (**mandatory**) | **OpenReview** → `https://openreview.net/group?id=MICCAI.org/2026/Challenge` |
> | 3 | Signed copyright form | organisers share it — ⚠️ the Instructions wiki says "at camera-ready", the Timeline says "at final submission". **Inconsistent — ask on the Community tab.** |
>
> ### 3. 📝 PAPER SPEC (full detail in **§12**)
> **8–10 pages excluding references**, **Springer LNCS**. Must report **(a) source-code GitHub link**,
> (b) method description, (c) **results on training + validation data**. Camera-ready = the identical paper
> plus test results. **Nothing in it waits on the test set — it can be written today from what we already have.**
> Must give our **Synapse team name (`3495063`)** at submission so paper ↔ Docker link. Mandatory citations
> (BraTS-MET + flagship + MedPerf) and the required acknowledgement string are in §12.
>
> ### 4. 🐳 DOCKER SPEC (full detail in **§13**) — the **12-hour total** limit is the one to measure
> A10G 24 GB · 16 vCPU · max CUDA **13.0** · `--memory 48G` · `--shm-size 16G` · **zero network** ·
> `/input` read-only · `/output` **flat**. **12 h for the entire hidden test set**, and our best is a
> **3-way fusion**. That is ~4 min/case at 179-ish cases for three models + hysteresis. **Untested. Measure it
> before anything else.** Organisers **do not verify containers until after the queues close** — a container
> that fails to run is simply not evaluated, with no second chance.
>
> ### 5. 📉 STANDING (fresh pull 07-24, 1,943 scored subs / 863 with 12 metrics, **71 teams**)
> **#7/71 excl-absent · #6/71 zero-for-absent · mr 60.4 · best still `9771992`** (unchanged since 07-16).
> **The 46.8 → 60.4 drift is pure field growth (§0 time-dependence), not regression** — our 12 scores are
> byte-identical and per-metric ranks moved in lockstep (DSC wt 9→33 for the same .725). Per-metric now:
> DSC **33/27/17/28** · NSD **38/46/36/49** · F1 **163/142/145** · **f1_rc rank 1** 🥇.
>
> | # | team | mr | best sub | DSC wt/tc/et/rc | F1 wt/tc/et/rc |
> |---|---|---|---|---|---|
> | 1 | 3588363 *(9 metrics, no RC)* | 11.8 | 9768192 | .728/.772/.738/— | **.662/.696/.664**/— |
> | 2 | 3589042 | 33.8 | 9772191 | .723/.767/.747/.671 | .399/.495/.489/.167 |
> | 3 | 3594706 | 36.3 | 9770847 | .720/.765/.745/.671 | .402/.507/.501/.167 |
> | 4 | 3590545 | 38.7 | **9772840** ← new | .720/.765/.745/.671 | .402/.507/.501/.167 |
> | 5 | 3595145 | 48.8 | 9770583 | .719/.764/.744/.671 | .402/.507/.501/.000 |
> | 6 | 3595169 | 50.4 | 9768659 | .719/.768/.748/.601 | .410/.517/.512/.111 |
> | **7** | **us · 3495063** | **60.4** | **9771992** | .725/.766/.746/.630 | .376/.478/.476/**.167** |
> | 8 | 3601728 | 106.3 | 9773198 | .721/.767/.738/.589 | .367/.475/.471/.000 |
> | 9 | 3604251 | 106.5 | 9773182 | .721/.769/.742/.594 | .363/.467/.462/.000 |
> | 10 | 3604144 | 106.8 | 9773185 | .726/.760/.741/.588 | .376/.484/.461/.000 |
>
> **3590545 climbed to #4** by swapping its rc .601 → **.671** (`9772840`). That `.671` + `.402/.507/.501`
> row now appears under **three** different teams — one recipe is circulating, and it matches our DSC while
> beating our tumour F1. **#8–10 are a cluster of near-clone newcomers (3601728/3604251/3604144/3604131),
> mr ~106–108, no threat.**
>
> ### 6. 🔴 WE WENT DARK 07-20 → 07-24 — ~10 SLOTS BURNED TO ZERO
> Last submission is `9772603` (07-19 14:48). Nothing since. Rivals submitted 50–60/day throughout.
> The cause was this document: it said the round closed 07-23 and that the paper was an August problem, so
> the endgame looked finished. **It wasn't.**
>
> ### 7. ✅ `9772603` (rc_step0125) SCORED — WORSE, RC STEP AXIS CLOSED
> `.727/.767/.747/**.622**` · `.376/.478/.476/**.000**`. It hit the tumour metrics byte-identically as
> designed, but RC DSC fell .630→.622 **and `f1_rc` collapsed .167 → .000 — the single metric we lead the
> field on.** Step size is closed for RC as it was for tumour. Do not revisit. (`sub_rc_s0125.zip` and
> `lb_snapshot.csv` are on sie271; `ccloss_s0125.log` is the 8 h predict run that produced it.)
>
> ### ⚠️ PRIORITY IS NOW INVERTED vs §6
> §6 ranks decode/F1 experiments first. **That ordering is dead.** With 6 days left the ranking is:
> **(1) Docker — it has a hard, unmeasured 12 h gate; (2) short paper — mandatory, writable today from
> existing results; (3) validation slots — a bonus, and every remaining F1 axis is already closed anyway (§1).**
> The scoring work is done and stable. The only things that can still cost us the *entire* submission are
> the two artifacts §0 told us to ignore.

> **⚡ 07-18 delta — every remaining DECODE + ENSEMBLE axis is now CLOSED; the RC step-0.125 probe is IN FLIGHT for 07-19.**
> 1. 🔴 **D015-UNION (both 07-18 slots) SCORED — NET WORSE, lever CLOSED.** cons `9772489` (MINSZ 15):
>    F1 **+.034/+.027/+.031** (biggest F1 lift since hysteresis — the grafts ARE real small mets) but DSC
>    −.010/−.012/−.012 → **mr 57.8 vs 9771992's 46.8**. aggr `9772490` (MINSZ 1): **mr 99.8**. Monotonic:
>    more grafts = worse. **§4c's "DSC preserved by construction" is FALSE for LESIONWISE DSC** (BraTS's
>    metric): every grafted small lesion enters the per-lesion DSC average below the mean, and that drop lands
>    in the PACKED DSC field while the F1 gain lands in the SPARSE field → net-negative (the §10a rank-density
>    law again). The union rides the same DSC↔F1 frontier as the seed probe; it does NOT escape it.
>    **§1c datapoint:** aggr's +156 sub-15-vox specks over cons buy only +.003/.004/.005 F1 → overwhelmingly
>    MIMICS, not real mets.
> 2. 🔴 **MIMIC/SURFACE FP FILTER (§1b/§1c) MEASURED + DROPPED — no slot spent.** `fp_surface_dist.py`: FP
>    centroid brain-surface-distance med **8.8mm** vs GT lesions **11.2mm**, IQRs overlap almost fully, **AUC
>    ≈ 0.575** (need ~0.70+ to filter). FPs are only marginally more peripheral — real grey-white-junction
>    mets sit at the same depth, so any surface filter kills real mets too. **§1c's mimic read was right by
>    eye but is not anatomically separable. The last DECODE-level F1 axis is closed.**
> 3. 🔴 **ENSEMBLING IS MAXED — no free wider fusion exists.** All 6 cached members are same ResEncL/D007
>    lineage → they CORROBORATE the FPs (§1: all nets fire on the same FP) → averaging more dilutes nothing.
>    Evidence: the original 28-decode search already tested a 4th member (`ccloss_v2`) at every blend weight +
>    veto/gate (none beat 3-way, `rank_fusions.py`); and today `score_4way_smallcc.py` scored **4-way+smallcc**
>    (harness-valid, pure average): **ΔF1 −.004/−.005/−.008**, does NOT dominate. `pred_ft` unusable (44/80,
>    6-ch BackSplit). **The ONLY remaining dilution lever is a DECORRELATED member** — fold-0 split or a
>    different PATCH SIZE (arch doesn't decorrelate; ResEncXL≡ResEncL), ideally trained with an **NCL /
>    orthogonality loss** (web-checked: the principled cure for corroborated FPs). That's a TRAINING move and
>    likely misses the 07-23 close on the 611MHz box.
> 4. 🟢 **07-19 PRE-STAGED = RC STEP-0.125 PROBE (IN FLIGHT).** With all F1 axes closed, the one untried lever
>    with real headroom is **RC** (our weakest ranks: DSC 26, **NSD 46**). Re-predicting the CC-loss donor
>    (`Dataset011` `nnUNetTrainerCCLoss_BackSplit_FT` chk best) on the 179 val at **step_size 0.125** (was
>    0.25) → finer boundaries on the LARGE cavities → targets **RC-NSD** specifically. `pred_ccloss_s0125` on
>    sie271, launched via `run_ccloss_s0125.sh` (setsid nohup, log `ccloss_s0125.log`), ~8-9h (540s/3-case
>    probe), ETA ~morning 07-19. **Then ONE COMMAND:** `bash ~/brats2025/finish_rc_lowstep.sh` (STAGED —
>    checks `.done`, assembles tumour BYTE-IDENTICAL to 9771992 with only RC swapped, verify-gate, zips
>    `sub_rc_s0125.zip`; does NOT submit). Submit = `./.venv/bin/python submit.py sub_rc_s0125.zip rc_step0125`.
>    Only the 4 RC metrics can move. **Board judges it (RC-NSD unmeasurable locally — hangs; `score_rc_native.py`
>    is gone).** If it helps, next rung = step 0.1 (~15× windows, slower). The RC THRESHOLD sweep (0.5→lower)
>    is the untried companion lever, also §5-safe (keep-largest → single component = pure extent).
> 5. **Standing unchanged (07-18 pull, 1,660 scored subs): #7/57 excl-absent · #6 zero-for-absent · mr ~47 ·
>    best still `9771992`.** rank12/lbquery2/ours all working on sie271 `~/brats2025/.venv` (confirmed today).
>
> **Carry-over 07-17 20:24 delta below (D015 scored, first F1 win — the donor that fed the now-closed union):**

> **⚡ 07-17 20:24 delta — `9772272` (D015) IS SCORED, and it is the first F1 win in weeks.**
> 1. 🟢 **D015 DilateISO standalone (`9772272`) LIFTED TUMOUR F1: .388/.492/.489 vs 9771992's
>    .376/.478/.476 — +.012/+.014/+.013 on all three cells** (§4b, §10a). DSC tanked (.661/.649/.615, it is a
>    lone unfused model, expected) so its own mean rank is poor — **but the F1 direction is the point.**
>    Dilate-training is the **first lever since hysteresis to move F1 the right way without riding the seed
>    frontier.** The +146 extra instances were mostly REAL, not §1c mimics.
> 1b. 🟢 **THE D015-UNION IS BUILT + STAGED (§4c) — 07-18 is pre-loaded.** Two zips on sie271 graft D015's
>    missed detections onto the byte-identical 9771992 base (so they CANNOT score below our floor): 
>    `sub_d015union_cons.zip` (222 new comps) and `sub_d015union_aggr.zip` (378). **Submit BOTH 07-18, read
>    F1** (DSC is preserved by construction). §5/§4c: component-count change → harness can't score it, board decides.
> 2. **Standing (fresh 20:24 pull, 1,628 scored subs): #7/57 excl-absent · #6 zero-for-absent · mr 46.3 ·
>    best still `9771992`** (§0a). D015 does not become the new best (its DSC hit sinks its mr); it is a
>    donor for a future fusion, not a submission on its own.
> 3. **Both 07-17 slots spent** (§10a): SLOT 1 = `9772250` `lo 0.20` (mr 46.50, extent optimum bracketed at
>    0.25 — do not push extent further); SLOT 2 = `9772272` D015 (this delta). **Next slots open 07-18.**
>
> **Carry-over from the earlier 07-17 delta (all still true):** SmallCC (D014) is DEAD — it did not un-blind
> small lesions (§4); D015 replaced it and has now trained, scored, and lifted F1 (above). The FP montage was
> INVALID (cx/cy swap), now fixed + re-read — the FPs look like MIMICS (vessels/dura/midline), not annotation
> gaps (§1b, §1c). The ranking tooling was never gone — it was split across boxes with a hidden `.venv`,
> rebuilt on sie271 (§0b). The RC ensemble is dead (`9771991`: RC .630→.628) and `hi` is optimal at 0.70;
> **`lo` (extent) is now also closed — bracketed at 0.25** (§10a). **After D015, F1 is the only open axis and
> the fusion-union of D015 is the live lever.**

---

## 0. STATE — 2026-07-14

**🏁 STANDING = #7 / 52 teams, mean rank ~56.7. Best = `9771576`** (= `9771508` + RC-fix removed; the two are within 0.2 mr = noise, treat them as equivalent).
DSC .720/.761/.737/.630 · NSD .739/.811/.798/.521 · F1 .376/.479/.477/.167

~~**Round closes 2026-07-23.**~~ ❌ **WRONG — the deadline is `2026-07-30 23:59 UTC` (07-24 delta §1).**
2 submission slots/day. **Both 07-14 slots USED:** `9771508` (3-way fusion, team best) + `9771576` (RC-fix removed — see §5).
~~**Paper + Docker are due in AUGUST, after results — NOT a July gate, not a DQ risk.** Do not spend July on them.~~
☠️ **THIS SENTENCE WAS FALSE AND IT COST US FIVE DARK DAYS.** Docker + short paper + copyright form are **all
due Jul 30**, and **no paper ⇒ the Docker is never even run ⇒ no final ranking.** The August dates are review
milestones, not the gate. **See the 07-24 delta at the top, plus §12 (paper) and §13 (Docker).**

**⚠️ RANKING USES 12 METRICS, NOT 11.** The old handoff dropped `small_instance_f1_rc`. It IS scored (598/1458 subs), and **we lead the field on it (.167, tied with 6 teams)**. Including it improves us (we lead that metric). Position is #7 either way, but the top-6 order shuffles. **Ranking scripts must use all 12.**

**⚠️ MEAN RANK IS TIME-DEPENDENT.** It is a rank against every scored submission, so it *inflates for everyone* as the field grows. Numbers quoted on different days are NOT comparable. Only ever compare submissions **within one snapshot** (`lbquery2.py` → `rank12.py`). Our mr has drifted 45 → 57 this week without our scores changing at all.

### 0a. ✅ LIVE STANDING — fresh pull 2026-07-17 20:24 (tooling REBUILT, see §0b)

**#7 / 57 teams (excl-absent) · #6 / 57 (zero-for-absent) · mean rank 46.3 · best = `9771992`.**
**Both of today's slots are SPENT** (`9772250` lo 0.20, `9772272` D015 — §10a). 1,628 scored subs in the table.
Per-metric ranks: DSC **9 / 17 / 16 / 26** · NSD **11 / 26 / 26 / 46** · F1 **133 / 120 / 125** · **f1_rc rank 1** 🥇.
The three tumour F1 cells remain the entire gap (§0 diagnosis holds). **D015 (§4b) is the first thing to dent
them — but only as a fusion donor, not standalone.** Zero-for-absent top of board: 3589042 25.2 → 3594706 25.5
→ 3595145 35.9 → 3590545/3595169 38.9 → **us 46.3** → 3588221 92.4. The 25–39 cluster is still the real target.

> _Prior pull for reference (14:00, 1,615 subs, mr 45.1): scores identical, only the field moved — the §0
> time-dependence again. Never read the 45.1 → 46.3 drift as us regressing; our per-metric ranks are unchanged._

> **Re-pulled 34 min later (14:34) — a live demo of the §0 time-dependence.** Identical scores, identical
> per-metric ranks, still #7. Only the field moved (1,615→1,617 scored, 684→685 with metrics) and **our mr
> drifted 45.1 → 45.3 for free**. Rivals drifted in lockstep (3589042 24.2→24.5, 3594706 24.8→24.9) — that
> is *not* them improving. **A 0.2 mr change over half an hour is pure field growth. Never read it as signal.**

| # | team | mr | best sub | DSC wt/tc/et/rc | F1 wt/tc/et/rc |
|---|---|---|---|---|---|
| 1 | 3588363 *(no RC, 9 metrics)* | **6.1** | 9768192 | .728/.772/.738/ — | **.662/.696/.664**/ — |
| 2 | 3589042 | 24.2 | **9772191** ← new | .723/.767/.747/**.671** | .399/.495/.489/.167 |
| 3 | 3594706 | 24.8 | 9770847 | .720/.765/.745/**.671** | .402/.507/.501/.167 |
| 4 | 3595145 | 35.0 | 9770583 | .719/.764/.744/**.671** | .402/.507/.501/.000 |
| 5 | 3590545 | 37.7 | 9768754 | .719/.768/.748/.601 | .410/.517/.512/.111 |
| 6 | 3595169 | 37.7 | 9768659 | .719/.768/.748/.601 | .410/.517/.512/.111 |
| **7** | **us · 3495063** | **45.1** | **9771992** | .725/.766/.746/.630 | .376/.478/.476/**.167** |
| 8 | 3588221 | 89.8 | 9771758 | .725/.760/.732/.586 | .368/.475/.470/.000 |

**Our per-metric ranks (of the subs carrying each metric):** DSC **7 / 16 / 15 / 26** · NSD **10 / 25 / 26 / 46**
· F1 **130 / 117 / 122** · **f1_rc rank 1** 🥇. **The three tumour F1 cells are still the entire gap** — the
§0 diagnosis holds, only harder: we are top-26 on all 8 DSC/NSD metrics and ~120th on tumour F1.

**Under ZERO-FOR-ABSENT we are #6** and 3588363 collapses to #9 (mr 116.2) — its 6.1 is bought entirely by
being ranked on **9 metrics, not 12**. Its DSC is ordinary (.728/.772/.738, ~ours); it leads on F1 alone.

⚠️ **Rivals are still moving:** 3589042 took #2 with a brand-new `9772191` (rc .671). Nobody is idle.

### 0b. ✅ THE RANKING TOOLING — WAS gone, REBUILT 2026-07-17

**It was never fully lost — it was split across two boxes, and `.venv` is hidden so a `~/brats2025/*` glob
misses it.** For the next agent:
- **`sie271`: `~/brats2025/.venv`** — synapseclient **4.13.0** + the `synapse` CLI. **No pandas** (so
  `tableQuery(...).asDataFrame()` throws — read `q.filepath` with the `csv` module instead).
- **`sie272`: `~/brats2025/syn.tok`** + `syn_check.py`. The token matches §9.
- ❌ `sie209-lap` (the laptop, i.e. the documented submission box) has **neither**. §7 is wrong about this.

**Now unified on sie271:** `syn.tok`, `syn_check.py`, **`lbquery2.py`**, **`rank12.py`**, `ours.py` — all in
`~/brats2025/`, all rebuilt/restored and working. Run:
```bash
cd ~/brats2025
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring ./.venv/bin/python lbquery2.py   # -> lb_snapshot.csv
./.venv/bin/python rank12.py     # 12-metric board, both conventions, our per-metric ranks
./.venv/bin/python ours.py       # our 42 scored subs by date + slots used per day
```
`syn_check.py` confirms auth (`LOGIN_OK user=aymuos id=3495063`). The keyring null is **required** headless
(§5). `getSubmissions()` still 403s for participants — expected, ignore it.

⚠️ **`rank12.py` is a REBUILD, not the original — its tie convention differs.** It gives tied submissions the
same (better) rank, so our `f1_rc` .167 reads **rank 1**, where the old script called the same score **rank 19**.
Ordering and team set reproduce the 07-14 html exactly (validating it in the §2 sense), but **absolute mr is
not comparable to the old numbers** (45.1 here vs 46.5 in the html for a *different* best sub). Only compare
within one `lb_snapshot.csv`.

**`lbquery2.py` and `rank12.py` do not exist**, on any box: not `sie209-lap`, not `sie271-pc`, not `sie272-pc`.
Nothing anywhere references the scores table (`syn74508245`) or the queue (`9619537`). **`synapseclient` is
also not installed on `sie209-lap`** — which contradicts §7, since sie209-lap *is* the submission box.

**Consequence: the standing cannot be refreshed, and has not been since 07-14.** Rebuilding it means
reinstalling `synapseclient` and rewriting both scripts — including `rank12.py`'s **12-metric** logic
(§0: `small_instance_f1_rc` must be in, it is the metric we lead). Budget ~1 h. Do this before any
claim about where we stand, and **do not compare a fresh pull against the numbers below.**

**Last valid snapshot = the 07-14 pull frozen into `brats2026final.html`** (1,486 scored subs; all mean
ranks there are mutually comparable, and its mr for us is **46.5**, not the 56.7 quoted above — same
submission, different pull; this is the time-dependence, not a discrepancy). Top 8 from that snapshot:

| # | team | mr | best sub | DSC wt/tc/et/rc | F1 wt/tc/et/rc |
|---|---|---|---|---|---|
| 1 | 3588363 *(no RC)* | 5.3 | 9768192 | .728/.772/.738/ — | **.662/.696/.664**/ — |
| 2 | 3594706 | 20.8 | 9770847 | .720/.765/.745/**.671** | .402/.507/.501/.167 |
| 3 | 3589042 | 21.3 | 9770160 | .722/**.772**/**.752**/.601 | .410/.517/.512/.111 |
| 4 | 3595145 | 28.5 | 9770818 | .719/.757/.738/**.671** | .403/.514/.508/.000 |
| 5 | 3590545 | 30.8 | 9768754 | .719/.768/.748/.601 | .410/.517/.512/.111 |
| 6 | 3595169 | 30.8 | 9768659 | .719/.768/.748/.601 | .410/.517/.512/.111 |
| **7** | **us · 3495063** | **46.5** | 9771508 | .720/.761/.737/**.630** | .376/.479/.477/**.167** |
| 8 | 3587408 | 88.7 | 9771376 | .712/.748/.726/.582 | .386/.472/.472/.000 |

Two things to notice. **#1 scores no RC at all** (empty, not zero) and leads purely on F1 (~1.5× ours) at a
DSC essentially equal to ours — under a zero-for-absent-RC convention we are already #6. And **rows 5 and 6
are byte-identical on all 12 metrics** from two different team IDs — that is one pipeline submitted twice,
and rows 3/5/6 share an F1 triple, so a common recipe is circulating near the top. The real target is the
**28–31 cluster (ranks 4–6)**, not #1.

### The one table that matters

Our rank on each metric, of ~600 scored submissions:

| metric | ours | field best | % of best | rank |
|---|---|---|---|---|
| DSC wt/tc/et/rc | .720/.761/.737/.630 | .728/.772/.752/.671 | 94–99% | 20/25/40/22 |
| NSD wt/tc/et/rc | .739/.811/.798/.521 | .770/.829/.818/.556 | 94–98% | 11/28/50/41 |
| F1 rc | .167 | .167 | **100%** | 19 |
| **F1 wt** | **.376** | **.662** | **57%** | **114** |
| **F1 tc** | **.479** | **.696** | **69%** | **99** |
| **F1 et** | **.477** | **.664** | **72%** | **102** |

**On 9 of 12 metrics we are a top-50 submission at 94–100% of the best in the competition.** The three tumour F1 cells are the entire gap. Being the single best segmenter in the field on all 8 DSC/NSD metrics would only reach **#4**; simply having `9771380`'s F1 (which we already produced) would reach **#2**.

---

## 1. 🔴 THE CENTRAL FINDING — read this before proposing anything

Measured 2026-07-14 on 80 GT cases with the official metric:

| | |
|---|---|
| **median peak prob inside our FALSE POSITIVES** | **0.96** |
| median peak prob inside lesions we MISS | 0.55 |
| small lesions we detect | 45% |
| **misses the model is BLIND to** (peak < 0.10) | **43%** |
| **FPs that are genuine hallucinations** (nowhere near GT) | **86%** |

**The model is more confident about its mistakes than about the lesions it finds.**

Our false positives are **confident** (0.96), **spatially coherent** (they survive Gaussian smoothing — solid blobs, not noise), **isolated** (86% nowhere near real tumour), and **corroborated** (they survived averaging across three independently-trained networks; all three fire on them).

### ⛔ THIS CLOSES THE DECODE. Seven mechanisms refuted, all for the same reason:

| tried | result |
|---|---|
| small-component filters (9768934, 9769148) | annihilated F1. small IS real |
| FP-rejection by mean confidence (9770850, 9770859) | recovered DSC, killed the recall it was meant to keep |
| **CC-loss v2** — per-component FP penalty in the loss | FP −53% but **TP −24%**. Collapsed into **global under-confidence**: asked to suppress things it believed in, its only move was to believe less in everything |
| **v2 as a learned veto** | at t=0.5 it rejected *everything* (TP 1.82 → 0.51). It never learned what an FP looks like, because they look like lesions |
| **v1/v2 blend** | removes FPs and TPs in lockstep all the way down. No sweet spot exists |
| **size-adaptive seed** | reproduces the plain global threshold exactly. Borderline components are all small anyway |
| **scale-space seed** (Gaussian-smoothed seed) | at every matched FP rate, the plain threshold detects MORE. A confident coherent blob survives blurring |
| **fragment healing** | only **14%** of FPs are healable fragments. 86% are genuine |

**Every one of these separates on confidence, or something monotone in it. Confidence is INVERTED here. Do not propose another FP-suppression idea without new information.**

### The two live explanations
- **Mimics** — enhancing vessels, choroid plexus, dural structures. A radiologist excludes them with anatomical context the network lacks. Fixable, hard.
- **Annotation gaps** — some "FPs" may be **real metastases the annotators missed**. Confident, coherent, isolated, lesion-shaped, agreed on by 3 nets. **If so, part of the F1 ceiling is not ours to fix.**

**🔍 CHEAPEST UNRUN EXPERIMENT: crop T1c around the 50 most confident FPs and LOOK AT THEM.** An afternoon. Decides whether the remaining week is worth spending on detection.

### 1b. 🔴 THE FP MONTAGE EXISTS AND IS INVALID — do not read it (found 2026-07-17)

`~/brats2025/fp_crops.py` was written and run on **07-15 08:35** (48 TC FPs found, peak 1.000..0.706;
output `fp_montage.png` + `fp_top.json` on sie271). **The handoff predates it, so no verdict was ever
recorded — and the montage it produced cannot support one.**

**The bug** (`fp_crops.py`, in `collect()`):
```python
cy, cx = [int(round(c)) for c in center_of_mass(m.max(axis=2))]   # cy=axis0, cx=axis1
...
x0, x1 = max(0, cx - R), min(t1c.shape[0], cx + R)                # indexes axis0 WITH cx
y0, y1 = max(0, cy - R), min(t1c.shape[1], cy + R)                # indexes axis1 WITH cy
```
The unpack names axis0 `cy` and axis1 `cx`; the slicing then indexes axis0 with `cx` and axis1 with `cy`.
**They are swapped**, so every crop is taken at the transposed location — usually outside the brain,
which is why ~10 of the 48 panels are pure black with a red circle floating in the void.

**Verified empirically** on the top FP (`BraTS-MET-00647-000`, stored cx,cy,cz = 88,125,44):
```
pred[cx, cy, cz]  (as the script indexes) = False   <- misses the FP entirely
pred[cy, cx, cz]  (swapped)               = True    <- lands inside the FP component
```
⚠️ **The montage looks plausible and that is the trap.** `x0 = cx - R`, so the red marker is drawn at the
crop centre *by construction* — it is dead-centre in all 48 panels whether or not the crop is right. The
circle carries **zero** information about where the FP is.

**✅ FIXED + RE-RUN 2026-07-17 13:56.** `fp_crops.py:61` now unpacks `cx, cy` (axis0 = x, matching the
slicing); backup at `fp_crops.py.bak`. Re-ran clean (same 48 FPs, peak 1.000..0.706), `fp_montage.png` +
`fp_top.json` on sie271 are now **valid**. Every crop lands on tissue; no black panels.

### 1c. 👁️ WHAT THE FPs ACTUALLY LOOK LIKE — first look, 2026-07-17

⚠️ **Read this as a non-radiologist's read of 48 single 56×56 axial T1c slices, not a verdict.** It wants
confirmation from someone who reads brain MR. But the pattern is strong and consistent.

**The FPs are overwhelmingly small bright enhancing foci in three anatomically-suspicious places:**
1. **At the brain surface / cortical margin / dura** — a large share of the 48 sit right on the rim, often
   with a bright dural line running through the crop (e.g. 00653 p1.00 67v, 00636 p0.95 23v, 00710 p1.00 19v).
2. **On or beside bright linear/tubular structures = vessels** — several circles sit directly on a bright
   line running through the crop (00747 p1.00 61v is the clearest; also 00756, 00149 p0.98 28v, 00644, 00114).
3. **At the midline** — falx / superior sagittal sinus territory (00226 p1.00 39v, 00217 p0.99 27v, 00653).

**This is the MIMIC hypothesis (§1), not the annotation-gap hypothesis.** Vessels, dura, and midline sinus
are exactly the list §1 named. Critically, **almost none look like a classic annotation gap would** — an
isolated round parenchymal met sitting in white matter with nothing else around it. The handful that do
(00408 p0.87 17v, 00408 p0.79 15v, 00217 p0.96 68v, 00691 p0.77 13v — circles in fairly uniform white
matter) are the real annotation-gap candidates, and they are **a minority, not the 86%**.

**Why this matters:** §1 concluded "confidence is inverted, do not propose another FP-suppression idea
**without new information**." *This is the new information.* The FPs are not confidence-separable — but
they may be **anatomically** separable, and that is a different axis entirely. Vessel/dura/surface
proximity is context the network lacks and a radiologist uses. It also means the F1 ceiling is probably
**ours to fix after all** — the pessimistic "we're fighting annotation gaps" branch of §1 looks unlikely.

**🔴 BEFORE building anything on this, two hard constraints:**
- **Quantify it first, don't trust my eyeballs.** Measure distance-from-brain-surface (and if possible a
  vessel proxy) for FP centroids vs GT lesion centroids. If FPs are systematically peripheral and GT mets
  are not, the signal is real and measurable. If they overlap, this read is wrong. **Cheap — do it first.**
- ⚠️ **A surface/vessel FP filter REMOVES COMPONENTS → §5 says the harness CANNOT tune it.** Memorisation
  suppresses exactly the FPs such a rule would admit or remove, so local numbers will lie about it — the
  same trap that produced the wrong-sign RC-fix call (`9771576`). This must be validated on the leaderboard
  with real slots, or on a genuine held-out fold (§6 #5), **not** on the 80-case memorised set.

---

## 2. 🧰 THE LOCAL HARNESS — the thing that changed everything

Before it existed, F1 moved **+0.001 in ten days** of blind 2-slots-a-day guessing.

- **Official** `github.com/BraTS/BraTS_evaluation` + panoptica, `config_mets.yaml`. Cloned to sie271/sie272 `~/brats2025/brats_eval/`. venv: `~/brats2025/evalenv/`.
- **Tumour set:** 80 GT cases (`~/brats2025/localeval/`), drawn from the 267 training cases that contain a small lesion (cases with none return NaN for small-F1 and are *excluded* by the official parser — they're wasted compute).
- **Cavity set:** 100 GT cases (`~/brats2025/rceval/`), stratified by cavity volume.
- **VALIDATION GATE:** it reproduces the known leaderboard ordering of 4 scored submissions. If a change breaks that, throw the harness away.

### ⚠️ ITS LIMITS — respect these
- **Everything is `fold_all`. No held-out split anywhere.** The models were TRAINED on these cases.
- **Valid for: DECODES** (same probabilities, different post-processing — memorisation is common-mode and cancels from the ranking).
- **NOT valid for: comparing MODELS** or absolute scores. I violated this for the RC donor and got a result that contradicts the leaderboard.
- **Direction yes, magnitude no.** It predicted `9771508` would gain ~16 mean-rank points and dominate all 9 tumour metrics. Real gain: **~6**, and ET DSC/NSD both *regressed*. Thresholds tuned on memorised data are slightly too strict on unseen data.
- 🔴 **NEVER tune anything that changes HOW MANY COMPONENTS you keep.** On memorised cases the model rarely emits spurious components — it knows the answer — so a "keep more components" rule looks free (TP up, FP flat). On unseen data it emits them, and the rule admits them. **Memorisation suppresses exactly the FPs such a rule would let in, so the harness is structurally blind to the cost.** This cost us `9771576` (predicted RC-DSC +0.028, delivered **−0.016** — wrong sign). ✅ Fusion weights and thresholds are safe (they change values, not counts). ❌ Component filters, keep-rules, size cutoffs are NOT.

### 🐛 Gotchas that cost hours
- **panoptica's NSD/HD95 HANG on large cavities** (surface distance, 100k+ voxels). The cavity eval set is sorted by volume so it wedges at ~case 60 every time. Fix: `config_mets_nohd.yaml` (HD95 stripped), and for RC use `score_rc_native.py` (metrics computed natively to the official formula; my FP/matching logic was validated 4/4 against panoptica on synthetic cases).
- **`pkill -f <script>` over SSH kills its own shell** — the remote command line contains the script name. Use explicit PIDs.
- **Background jobs die when the SSH session closes.** Use `setsid nohup … & disown`.

---

## 3. WHAT'S RUNNING — verified 2026-07-17 20:24

**NONE of ours. sie271 GPU is IDLE** (D015 finished, predicted, submitted, scored — §4b). SmallCC dead (§4),
BlobDist gone. **Only the conv-free thesis (not ours) is training, on sie272.**

| box | job | ours? | state |
|---|---|---|---|
| **sie271** (GB10, **611 MHz** throttled) | **IDLE** — no nnUNet procs, no compute-apps (verified 20:24) | — | **D015 DONE.** Trained 150 ep (ckpts 07-17), predicted 179 val cases, decoded (`assemble_d015.py`), submitted `9772272`, **scored** (§4b). `checkpoint_best.pth` + val probs `val_infer/pred_d015_s025/` on disk. **Free box — use it for the D015-union fusion (§6 #2).** Model dir: `…/models/Dataset015_BraTSMetsDilateISO/nnUNetTrainerDiceTopK10BCE_FT150__nnUNetResEncUNetLPlans__3d_fullres/fold_all/` |
| **sie272** (GB10, 2400 MHz) | `nnUNetv2_train 12 -tr nnUNetTrainer**ConvFreeXHuge**` from `~/envs/test_nightly` (verified 20:24, still running) | ❌ **NO** | Still the same `ssk23` account, **different project (the conv-free thesis) — DO NOT KILL.** Still the only fast GPU. |
| **sie236** (RTX 4090 + 5090) | someone else's | ❌ no | — |
| — | **BlobDist** (`-tr nnUNetTrainerBlobDist_NoRC`) | — | ✅ **Gone** — no longer running anywhere. §6's "kill BlobDist" is **done**. |

**D014 SmallCC:** stopped, not running. `checkpoint_best.pth` / `checkpoint_latest.pth` still on disk under
`…/models/Dataset014_BraTSMetsSmallCC/…/fold_all/` if you ever want to score it properly.

**No scoring or assembly jobs are running.**

### ✅ D015 IS DONE AND SCORED — its result, and what's left to measure
**D015 answered its question on the leaderboard: tumour F1 rose +.012/+.014/+.013 (§4b).** So the 43%
blind bucket almost certainly shrank — the dilate-trained net is firing on small mets it used to miss.

**Still worth running for the record (optional, cheap):** re-run `~/brats2025/miss_diag.py` on the D015 val
preds and compare the `peak < 0.10` bucket against the current 43% — quantifies *how much* of the blindness
the target-size lever removed. Reminder: **do NOT judge D015 on pseudo-Dice/EMA** — that metric scored
CC-loss v1 at 0.917 while its real lesion-wise DSC was 0.327, and read .93/.93/.88/.86 for SmallCC which
still failed (§4). The leaderboard F1 is the only judge that has ever been right, and it already spoke.

---

## 4. ☠️ SmallCC — DEAD. It did not un-blind them. (verdict 2026-07-15)

> **STOPPED at ep 98/150** on 07-15 08:18, and replaced by D015 (§4b, built 07-15 08:41 — hours later).
> **Verdict, per the `build_d015.py` docstring:** *"SmallCC (D014) attacked this by SAMPLING small lesions
> 25x more; **it did not un-blind them**."*
>
> **The 20× sampling fix worked exactly as designed and bought nothing.** Everything in §4 below was
> verified true — patches centred on a small lesion went 1.28% → 25.75% — and the blindness did not move.
> That is the finding: **the network is not blind to small mets because it rarely sees them.** Seeing them
> 20× more often did not make it fire. Whatever causes the 43% blind bucket, starvation is not it.
>
> ⚠️ Its pseudo-Dice at ep 98 was a healthy **.9245/.9007/.8569/.8744** — right to the end. §3's warning
> held: the metric lied again.
>
> 🔎 **Caveat worth knowing:** I found no saved `miss_diag_smallcc` output — only the one-line conclusion in
> the docstring, apparently from an interactive run. The checkpoints are still on disk, so if the D015 result
> is ambiguous and you need to know *how* SmallCC failed, that diagnostic is re-runnable.

_Original §4, retained because the mechanism was verified and the negative result is only meaningful with it:_

**Dataset014_BraTSMetsSmallCC** + `nnUNetTrainerSmallCC_FT`.

**Why:** nnU-Net's sampler picks a random **KEY** from `class_locations`, then a random **VOXEL** from it. Small mets live inside the `(1,2,3)` key among every SNFH and large-lesion voxel, so a 30-voxel met is patch-centred **~0.3%** of the time — almost no gradient, whatever the loss. **In the training set small lesions (5,853) OUTNUMBER large ones (3,217) nearly 2:1** and were being starved. `FgOversample` (0.33→0.50) failed because it oversamples *foreground* — the wrong distribution.

**The fix:** a new key `(1,2,3,99)` holding **only** small-component voxels (<200 vox), **every lesion weighted equally** (6 voxels each regardless of size — the blob-loss principle applied to sampling). The sampler picks keys uniformly → it wins 1 in 4.

**✅ VERIFIED EMPIRICALLY** (replicated nnU-Net's `get_bbox` logic, 4,000 draws):
- patch centred on a small lesion: **D007 1.28% → D014 25.75% (20×)**
- all 356 new-key voxels land inside a real small component (0 bad)
- with `oversample_foreground_percent=0.5` → **~13% of ALL patches**, vs ~0.6% before

**Base = D007 4-region TopK** (`checkpoint_topk_snap`), deliberately **NOT** BackSplit — BackSplit demonstrably costs F1 (9769617 control: .27/.33/.32 vs 4-region .40/.50/.50). Loss untouched (Dice + TopK-BCE). **This changes what the network SEES, nothing else.**

### 🐛 THE TRAP THAT NEARLY VOIDED IT
**nnU-Net reads the preprocessed-data path AND the results path from `dataset_name` INSIDE THE PLANS JSON — not from `-d 14`.** I copied D007's plans verbatim, so it silently trained on **D007's original pkls** and wrote results under Dataset007. It would have run two days and told us nothing. **If you clone a dataset, patch `dataset_name` in the plans.**

**Build script:** `~/brats2025/build_d014.py`. **Verify:** `~/brats2025/verify_d014.py`.

---

## 4b. 🚀 D015 DilateISO — SCORED as `9772272`: FIRST F1 WIN IN WEEKS (2026-07-17 20:24)

**Dataset015_BraTSMetsDilateISO** + stock `nnUNetTrainerDiceTopK10BCE_FT150`. Build: `~/brats2025/build_d015.py`.
Trained (150 ep, ckpts 07-17), predicted on the 179 val cases, decoded + submitted as **`9772272`**.

> **🟢 THE RESULT (`ours.py`, 20:24): D015 standalone LIFTED TUMOUR F1.**
> | | DSC wt/tc/et/rc | F1 wt/tc/et/rc |
> |---|---|---|
> | 9771992 (3-way best) | .725/.766/.746/.630 | .376/.478/.476/.167 |
> | **9772272 (D015 standalone erode-k1)** | **.661/.649/.615/.630** | **.388/.492/.489/.167** |
> | Δ | −.064/−.117/−.131 (expected) | **+.012/+.014/+.013** |
>
> **The bet paid.** §4b predicted "+146 TC instances → higher F1 IF they are real, lower if they are §1c
> mimics." **F1 rose on all three tumour cells → a good share of those extra small-lesion detections are
> REAL.** This is the **first lever since hysteresis to move F1 the right way** — and unlike the seed axis
> (9771835: F1 +.04 but DSC −.017, net worse mr), it did NOT do it by riding the DSC↔F1 frontier; the
> dilate-trained model simply *finds more true small mets*.
>
> **⚠️ Do NOT submit D015 standalone as our entry.** Its DSC collapse (it is one unfused model, and DSC is
> where the packed field lives) sinks its own mean rank far below 9771992. **The value is as a FUSION DONOR.**
> **→ THE LIVE LEVER (§6 #2): mask-level UNION of D015's *new* detections into the 3-way — keep the 3-way's
> DSC, graft D015's extra true positives for F1. Do NOT prob-average the dilated maps (§4b: dilated probs
> cannot be averaged into the fusion).** This is a component-count-increasing change, so §5 applies: the
> harness CANNOT score it — it must be validated on a real leaderboard slot (or a held-out fold, §6 #5).

> **🔴 THE EROSION FINDING — the trap `build_d015` walked into, and how it was fixed (07-17).**
> The build DROPPED inference erosion, arguing the small-lesion oversize is "negligible for the tiny
> volumetric-DSC weight". **That is a GLOBAL-Dice argument, and Task 1 scores LESIONWISE Dice** (every
> lesion weighted equally). Measured on 80 GT cases (`erode_cost.py`): if the net learns the dilated
> target, small lesions score **lesionwise Dice ~0.20 with no erosion vs ~1.0 with it.** Erosion is
> mandatory, and it must be **selective** — `build_d015` dilated ONLY isolated small mets, so eroding
> large structures (never dilated) would wrongly shrink our strong DSC/NSD metrics.
>
> **How much to erode was MEASURED, not guessed** (`d015_extent.py`, matched small TC components vs a clean
> member): median D015/REF size ratio **3.31** (mean 4.22) — D015 *did* learn a dilation, but only ~**k=1**
> effective, well short of the k=2 it was trained on. Geometry on the median 216-vox comp: erode k=1 → ~84
> vox (true ~56, Dice ~0.80 ✓); erode k=2 → ~21 vox (OVER-eroded below true, Dice ~0.55 ✗). **So erode k=1,
> not k=2.** k=2 would have destroyed the very small lesions D015 exists to find.
>
> **Decode (`assemble_d015.py`, on sie271):** D015 tumour → hysteresis lo .30/hi .70 → **selective erode k=1
> on components <500 vox** → RC grafted from CC-loss (keep-largest). Result vs the 3-way best: same total
> tumour volume (10.37M vs 10.54M — erosion de-bloated it correctly) but **1099 TC components vs 953 (+146)**
> and 10 empty cases vs 14. **That +146 instances IS the bet:** more small-lesion detections → higher F1 IF
> they are real, lower F1 if they are the §1c mimics. The leaderboard F1 cell is the only valid judge (§2 —
> harness cannot compare models). Standalone because dilated probs cannot be averaged into the fusion.
>
> **Scripts (all on sie271 `~/brats2025/`):** `predict_d015_val.sh`, `d015_extent.py`, `erode_cost.py`,
> `assemble_d015.py`, `segstats.py`. Val probs cached at `val_infer/pred_d015_s025/` (179 npz).

_Original design notes below (still accurate on the training side):_

**Undocumented before 07-17** — built 07-15, after the last handoff revision.

**The idea:** SmallCC made the model *see* small lesions more often and failed (§4). D015 makes the **target
bigger** instead. Isolated small mets are dilated by **k=2**, so a 30-voxel met becomes ~4× the volume — more
gradient per lesion, and a spatially larger object for a 3D receptive field to latch onto. **No inference-time
erosion**; predictions come out ~2 vox oversized, negligible for instance-F1 and for small lesions' tiny
volumetric-DSC weight.

**Why it is safe** (pre-checked on the 80 GT cases, per the build docstring):
- Only **isolated** small mets are dilated (no other met within 2k voxels) → **zero instance merging**.
  ~100% of blind small mets are isolated at k=1, **~88% at k=2**.
- **Large structures untouched** → no large-structure DSC collateral.
- Shell painted **only** into background/SNFH (labels 0, 2) — never over another met, RC(4), or ignore(−1).
- ✅ **No component-count change at inference** → per §5 this is a decode-safe, harness-tunable change.

**It changes ONLY the target the loss sees.** Sampling (`.pkl class_locations`) and image data (`.b2nd`) are
symlinked from D007 unchanged; original lesion voxels remain valid foreground centres inside the enlarged
lesion. Loss untouched. FT from `checkpoint_topk_snap`, `num_epochs=150`, `initial_lr=1e-3`.

⚠️ **The §4 plans-json trap applies to any cloned dataset** — nnU-Net reads `dataset_name` from *inside*
the plans, not from `-d 15`. D015 is confirmed writing to its own `Dataset015_BraTSMetsDilateISO` results
dir, so it was patched correctly.

---

## 4c. 🟢 D015-UNION — the fusion that banks D015's F1 without D015's DSC loss (STAGED 2026-07-17 21:00)

**The problem §4b left open.** D015 standalone (`9772272`) lifts tumour F1 (+.012/+.014/+.013) but its DSC
collapses (.661/.649/.615) — it is one dilated, unfused model. And its inflated extent means it **cannot be
probability-averaged** into the 3-way (that is why it shipped standalone). So the F1 win was real but
unusable *as a submission*. **D015-union is how you keep the F1 and throw away the DSC loss.**

**The recipe.** Take the CURRENT BEST `9771992` (3-way fusion) as an untouchable base, and **mask-union only
D015's detections that the 3-way MISSED:**
```
base (WT/TC/ET region masks)  = the CACHED 9771992 seg (val_infer/sub_slot2_lo025), LOADED not recomputed
D015 region mask (per channel) = hyst(dp[ch], lo .30/hi .70, 26-conn) -> selective erode k1 on comps <500vox
graft                          = D015 components with ZERO voxel-overlap with base AND >= MINSZ voxels
union[region]                  = base | graft ;  paint WT->2, TC->1, ET->3 (region nesting is automatic)
RC (label 4)                   = REUSED from the base seg (already keep-largest) — untouched
```

**Why it is safe — three structural guarantees, not tuning:**
1. **Base is byte-identical to 9771992** because it is *loaded from the cached seg*, not recomputed. **The
   union can therefore NEVER score below our banked floor** — it can only add components, never remove or
   move one. Worst case = it equals 9771992.
2. **Zero instance-merging:** only components with *zero* voxel-overlap with the base are grafted, so no
   graft can touch/merge an existing 3-way instance. Verified by construction.
3. **DSC is preserved:** the base voxels and their labels are unchanged, and grafts are small isolated mets.
   So **read F1, not DSC — DSC is not the signal here.** RC is reused verbatim (27 rc_cases, identical).

**⚠️ BUT it CHANGES COMPONENT COUNTS → §5's hard limit applies: the local harness CANNOT score it.**
Memorisation suppresses exactly the FPs a "graft more components" rule would admit, so any local number lies
(this is the trap that produced the wrong-sign RC-fix, `9771576`). **The leaderboard F1 cell is the ONLY
valid judge.** The offline output below is a sanity check (counts + verify gate), never a performance forecast.

**The two staged variants — a one-knob bracket on MINSZ (min grafted-component size):**

| zip (on sie271 `~/brats2025/`) | MINSZ | new comps | new vox | cases grafted | empties (base=14) | verify |
|---|---|---|---|---|---|---|
| **`sub_d015union_cons.zip`** | 15 | **222** | 76,549 | 43/179 | 10 | ✅ PASS |
| **`sub_d015union_aggr.zip`** | 1 | **378** | 77,520 | 60/179 | 9 | ✅ PASS |

**What the bracket isolates:** aggressive adds **+156 components but only +971 voxels over conservative → those
extras average ~6 vox each.** The 222 substantial detections and the empty-case drop (14→10) are SHARED by
both. So the only thing the two zips differ on is: **do the tiny sub-15-vox specks help F1 or are they §1c
mimics?** Genuinely undecidable offline (§1: small IS real 4×; §1c: mimics are small bright foci) → the board
decides. If both beat 9771992 on F1, the aggr−cons delta *is* the small-speck real/mimic ratio, a free §1c
datapoint.

**Submit (07-18, both slots) — submit.py needs the SYNAPSE `.venv` (not the numpy decode env):**
```bash
cd ~/brats2025
./.venv/bin/python submit.py sub_d015union_cons.zip d015union_cons_minsz15
./.venv/bin/python submit.py sub_d015union_aggr.zip d015union_aggr_minsz1
```
(submit.py nulls the keyring itself; §5 recipe.) **Build scripts:** `assemble_d015_union.py` +
`build_d015_union.sh` on sie271. Rebuild with `bash build_d015_union.sh` — it sources `nnunet_env.sh` for the
numpy env, loads the cached base + D015 probs (~6 min for both variants, CPU-only, no GPU inference).

---

## 5. SUBMISSIONS — recipe & the current best

**`9771508` (team best) = 3-way fusion.** Chosen by the harness out of 28 decodes, at zero slot cost.
```
 TopK-4reg (D007)  ┐
                   ├─► mean of all three ─► hysteresis lo .30 / seed .70 ─► WT · TC · ET
 completion (D011) ┤
 CC-loss FT (D011) ┘
 CC-loss FT ──────────► RC channel ──► thresh .50 ──► [RC-fix] ──► RC
```
**Why it works:** averaging the blob-loss model in **dilutes its false positives** (1.66 → 0.85/case) while its corroborated detections survive, because the other two vote for them. **Dilution is the ONE mechanism that has never failed.**

**Locked defaults:** step size **0.25**, TTA on, 4-region decode order WT(2)→TC(1)→ET(3)→RC(4).

### ⚠️ RC-FIX: the harness said "remove it", the leaderboard said "wash". READ THIS.

**Local (100 GT cavity cases):** keep-largest → keep-every-component-≥50vox gave **TP 0.96 → 1.05, FP flat at 0.01, RC-DSC .712 → .740 (+0.028)**. Looked like a free win. Shipped as `9771576`.

**Leaderboard reality (`9771576` vs `9771508`, tumour byte-identical):**
| | 9771508 | 9771576 | Δ rank |
|---|---|---|---|
| RC DSC | .630 (rank 28) | **.614** (rank 42) | **+14 WORSE** |
| RC NSD | .521 (rank 50) | **.534** (rank 38) | **−12 BETTER** |
| mean rank | 56.9 | 56.7 | −0.2 = **noise** |

**The harness got the SIGN wrong on RC-DSC.** Net: a wash. Either submission is fine.

#### 🔴 THE LESSON — a NEW limit on the harness, not in §2 before

**A decode that changes HOW MANY COMPONENTS YOU KEEP cannot be tuned on memorised data.**

On training cases the model rarely emits a spurious second cavity — it knows the answer. So "keep every component" looks free: TP rises, FP stays at 0.01. **On unseen data the model DOES emit spurious components**, and the same rule admits them. **Memorisation suppresses exactly the false positives the rule would let in, so the harness is structurally blind to the cost.**

That is NOT common-mode, and it breaks the "harness is safe for decodes" claim in the specific case where the decode alters component count.

- **3-way fusion WORKED** (averaging does not change component count) ✅
- **keep-≥50vox FAILED** (it does) ❌
- This also retroactively explains why the **local RC donor comparison contradicted the leaderboard**.

**Rule of thumb: tune thresholds and fusion weights locally. Do NOT tune anything that changes the number of predicted components.**

**Still untested: ensembling the cavity** — mean(CC-loss, FgOversample) gave RC-DSC .751 locally. **But that is a donor/model comparison AND it may shift component counts — treat as unreliable.** FgOversample val probs at step 0.25 do not exist (~2 h to predict).

**⚠️ RC-NSD is UNMEASURED locally.** panoptica AND a native implementation both hang on the biggest cavities (2 surface-distance transforms per component; the eval set is sorted by volume so it wedges at ~case 75). Notably, NSD is the one that *improved* on the leaderboard — we could not have predicted that either way.

### The DSC↔F1 frontier is REAL — do not try to reweight
Mean rank along the line between `9771508` (CC-loss weight ⅓) and `9771380` (weight ½):

| CC-loss weight | DSC wt/tc/et | F1 wt/tc/et | mean rank |
|---|---|---|---|
| **⅓ (9771508)** | .720/.761/.737 | .376/.479/.477 | **45.8** ← optimum |
| 0.42 | .695/.738/.714 | .403/.504/.510 | 82.7 |
| ½ (9771380) | .670/.715/.691 | .429/.530/.543 | 125.2 |

**Monotonically worse in every direction away from `9771508`.** Rank *density* is why: we sit at rank 20–40 on DSC where the field is packed (0.005 DSC = dozens of places), and rank ~100 on F1 where it's sparse. `9771508` is provably optimal on that line. The #2 finish exists only at an unreachable corner.

### Submit
```python
import os; os.environ["PYTHON_KEYRING_BACKEND"]="keyring.backends.null.Keyring"
import synapseclient; from synapseclient import File
syn=synapseclient.Synapse(silent=True, skip_checks=True); syn.login(authToken=TOK)
f=syn.store(File(ZIP, parent="syn75900400"))
sub=syn.submit(evaluation="9619537", entity=f, name="…")
```
Synapse: user `aymuos` id **3495063** · project `syn75900400` · queue **9619537** · scores in **`syn74508245`**. Token in §8.
**Verify gate:** 179 masks · labels ⊆ {0..4} · no label >8% of volume.
**Gotchas:** keyring must be nulled headless; `syn.getSubmissions()` → 403 for participants (just store+submit); Synapse READ_ONLY windows return 503.

---

## 6. NEXT — in priority order

> ## ⛔ 07-24: THIS ORDERING IS DEAD. DO NOT WORK THE LIST BELOW FIRST.
> It was written 07-17 against a **07-23 close** that does not exist and an "August paper" that is due
> **Jul 30**. The real order, with 6 days left:
> 1. 🐳 **DOCKER (§13)** — hard **12 h total** inference gate, **never measured**, and organisers do not test
>    it until after the queues close. Highest risk, longest lead time.
> 2. 📝 **SHORT PAPER (§12)** — **mandatory**; without it the Docker is never run. 8–10 pp LNCS, needs a
>    **public GitHub repo** (we have none). Fully writable today — nothing in it waits on the test set.
> 3. 🎯 **Validation slots** — a bonus now. Every decode/ensemble/F1 axis is closed (§1, 07-18 delta), and
>    the leaderboard closes Jul 30 with the rest. The only §5-safe untried point is `lo 0.25 / hi 0.65`,
>    which needs no new inference. **Do not let it displace 1 or 2.**
>
> _Below is the 07-17 list, kept for the evidence it records — not as a plan._

_(rewritten 2026-07-17; written against the since-superseded 07-23 close)_

1. **✅ DONE — LOOKED AT THE FALSE POSITIVES.** Bug fixed, montage re-run and read (§1b, §1c). **Verdict:
   they look like MIMICS — vessels, dura, surface, midline sinus — not annotation gaps.** The detection
   ceiling is probably ours to fix. **→ new #1 below.**
1b. **🔴 QUANTIFY THE MIMIC READ.** Distance-from-brain-surface (+ vessel proxy) for FP centroids vs GT
   lesion centroids. Turns a visual impression into a measurement, and it is cheap. **Do this before
   building anything on §1c.** If FPs are systematically peripheral and GT mets are not → the first
   genuinely new FP-separation axis since §1 closed the confidence route. ⚠️ But see §1c: any filter built
   on it **changes component counts**, so §5 forbids tuning it on the harness — leaderboard or held-out only.
2. 🟢 **✅ D015 SCORED — and it lifted F1 (§4b). NOW BUILD THE FUSION.** D015 standalone gained tumour F1
   +.012/+.014/+.013 over 9771992 → dilate-training does find more real small mets. **The next move is to
   graft D015's new detections into the 3-way by MASK UNION** (not prob-average — dilated maps cannot be
   averaged): run the 3-way fusion decode, then add any D015 component (post erode-k1) that does not already
   overlap a 3-way component. This keeps the 3-way's strong DSC while importing D015's F1. **⚠️ It CHANGES
   component counts → §5 forbids harness scoring → it costs a real slot (or a held-out fold, §6 #5).** Tune
   the union threshold (min D015 component size / peak) offline for sanity, but the F1 cell is the only judge.
   _Blindness note (§3): D015 was the target-size lever against the 43% blind bucket; F1 rose, so at least
   some of the blind small lesions are now being fired on. Re-run `miss_diag.py` on the D015 val preds to
   confirm the `peak<0.10` bucket shrank — but the leaderboard F1 already says the lever worked._
3. **Rebuild the ranking tooling** (§0b) — `synapseclient` + `lbquery2.py` + `rank12.py` (12 metrics). ~1 h.
   We are flying blind on the standing and have been since 07-14, with 6 days left.
4. **✅ 07-17 SLOTS BOTH SPENT + BOTH SCORED.** Slot 1 = `9772250` lo 0.20 (mr 46.50, extent optimum
   bracketed at 0.25 — §10a). Slot 2 = `9772272` D015 standalone erode-k1 — **F1 rose +.012/+.014/+.013,
   DSC fell as expected (§4b).** The question it answered (does dilate-training lift tumour F1?) is **YES.**
   **🟢 → 07-18 IS PRE-STAGED: submit the two D015-UNION zips (built 07-17 21:00, on sie271 `~/brats2025/`).**
   Both keep the exact cached 9771992 seg as base (byte-identical, so they CANNOT score below our banked floor)
   and mask-union only D015's *non-overlapping* new detections. Both passed the verify gate (179 masks, 0 bad
   labels, 0 RC-dominating). Submit with `submit.py <zip> <name>`:
   - **`sub_d015union_cons.zip`** (conservative, MINSZ 15): grafts **222** new comps / 43 cases, empties 14→10.
   - **`sub_d015union_aggr.zip`** (aggressive, MINSZ 1): grafts **378** new comps / 60 cases, empties 14→9.
   - **The bracket:** aggressive's +156 comps are ~6 vox each (near-zero volume) → they isolate one question,
     *do the tiny specks help F1?* The 222 substantial detections + the empty-case drop are shared by both.
     **⚠️ Component-count change → §5 says the harness CANNOT score these — the board is the only judge.**
     Read F1 (not DSC — base DSC is preserved by construction). Whichever wins becomes the fusion recipe;
     if both beat 9771992 on F1, aggressive-vs-conservative tells us the small-speck real/mimic ratio.
   **Build scripts:** `assemble_d015_union.py` + `build_d015_union.sh` on sie271. **Safe fallback if you'd
   rather not spend both slots on the union:** `lo 0.25 / hi 0.65` — the frontier at the new extent floor.
5. ~~**Predict FgOversample on val @ 0.25** → unlocks the RC ensemble (+0.011).~~ ❌ **ANSWERED — DROP IT.**
   `9771991` shipped the RC ensemble on 07-16: RC DSC **.630 → .628**. It does not help. §5 called it
   unreliable (donor/model comparison + component-count shift) and was right. The local **.751** was a lie.
5. **A model with a real held-out fold.** Everything is `fold_all`; the contamination is why forecasts are 3× optimistic and why the RC donor result is untrustworthy. One fold-0 run fixes the harness permanently and gives a decorrelated ensemble member.

✅ **Done / moot since 07-14:** ~~Kill BlobDist~~ (gone, §3) · ~~Move SmallCC to sie272~~ (SmallCC is dead, §4)
· ~~Score SmallCC~~ (failed, §4).

### ⛔ LOCKED REJECTS — evidence-backed, do NOT retry
- **RankSEG** (MONAI PR #8908, evaluated 2026-07-17) — an inference-time decoder maximising expected
  **samplewise Dice/IoU**. **Rejected, wrong axis.** (a) It targets overlap, where we are already rank
  7/16/15/26 DSC and 10/25/26/46 NSD; our whole deficit is F1 (rank ~120), and §0 caps a perfect DSC/NSD
  sweep at **#4** anyway — on DSC wt we are .725 vs a field best of .728, so there is no headroom to win.
  (b) Its **+0.0209 is measured against argmax**, and we have not argmaxed since June — we decode 4-region
  sigmoids with tuned hysteresis (worth −64 mr on its own, §10). Gains over a naive baseline do not transfer.
  (c) BraTS scores **lesionwise** DSC; RankSEG optimises **samplewise** Dice — different objective.
  (d) §1 predicts it **backfires on F1**: it keeps voxels by confidence, and our FPs peak at **0.96**, higher
  than the lesions we miss (0.55). (e) It is an adaptive per-sample **set-size** chooser → **changes component
  counts** → §5 says the harness cannot measure it, so it would cost real slots to evaluate. (f) The PR is
  **open/unmerged**, an optional external dep, n=20 on MSD Pancreas with 2/20 regressions. The 145 ms vs
  1.65 ms overhead is **not** the objection (~26 s over 179 cases) — the objective mismatch is.
  *(Steelman, for the record: it is a principled **per-image adaptive extent** chooser, and extent IS our live
  axis — 9771992. But we tune extent directly with a global `lo`, validate on the leaderboard, and spend no
  slots. Do the `lo 0.20` probe instead.)*
- **All FP suppression by confidence** (§1). Seven mechanisms, one refutation.
- **CC-loss v3 / peak-based FP penalty** — I proposed it hours before the diagnostic refuted it. FPs have peak 0.96; penalising peak = v2 all over again. *(The **seed-loss** half survives: force one voxel above `hi` inside every GT lesion. It attacks the blindness, not the FPs.)*
- **Reweighting the CC-loss ensemble** — the frontier is real and `9771508` is on the optimum.
- **Small-component / size filtering** — annihilates F1, four times.
- **Union recall recovery** (9770317), **re-thresholding** (9770029/30), **LoG hint channels** (9770485, mr 424).
- **Specialisation** — tumour-only (9770998) and cavity-only (9771148) models were both *worse* than the general net at their own speciality.
- **Distance/boundary loss** (9769328, mr 343) — and BlobDist's signed-DT term was **~2,400× too weak** to bite (per-voxel normalisation vs per-component recall).
- **MedNeXt-v2** (~3386 s/ep), **KiU-Net**, **U-Mamba**, **larger kernels**.
- **"ResEncXL" == "ResEncL"** for this dataset (identical net; the planners differ only in memory target). Arch is NOT a decorrelation lever — **patch size is**.

---

## 7. INFRASTRUCTURE

| host | role | GPU | notes |
|---|---|---|---|
| `sie209-lap` (laptop) | orchestration, **submission** | — | ⚠️ **`synapseclient` is NOT installed here** (verified 07-17) — reinstall before submitting. Submission zips. Also **no ranking scripts**, see §0b |
| `sie272-pc` | **training + inference** | GB10 @ **2400 MHz** | the fast one. `~/envs/brats` |
| `sie271-pc` | training + inference | GB10 @ **611 MHz** | **power-throttled, ~4× slower — NOT fixable** (shared Grace+Blackwell power envelope; controls read `[N/A]`). 12–14 W under load is NORMAL here, not a hang. Raw zips + `extracted/` live here |
| `sie236` | occasional | RTX 4090 + **5090** | log in as **`tosin`**. nnU-Net 2.8.1 env is **broken** (torchvision ABI). Built our own: `~/brats/ssk_env` (torch 2.11+cu128, nnU-Net **2.6.4**, torchvision from the cu128 index). Env: `source ~/brats/ssk_env.sh`. **Disk 99% full.** |

- **Sparks env:** `source ~/brats2025/nnunet_env.sh` → `nnUNet_raw/preprocessed/results` under `~/projects/CAI4CAI/SegData` + `/models`. Always `export nnUNet_compile=0`. nnU-Net **2.6.4**.
- **Inter-Spark fabric (ConnectX, ~200 MB/s):** sie271 `169.254.88.72`, sie272 `169.254.47.129`. **A key had to be generated** (`~/.ssh/id_ed25519_fabric` on sie272, authorised on sie271) — the old handoff assumed one existed and it didn't. sie236 cannot reach the Sparks: **relay through the laptop**.
- **GB10 gotcha:** `nvidia-smi` reports memory as `[N/A]`. Judge health by epoch time, not GPU%.
- 🐛 **sie271 does NOT absorb a heavy CPU job alongside training** (learned the hard way, 07-17). Running
  `fp_crops.py` (3 × `.npz` prob loads per case × 80 cases) at 13:50–13:56 was followed immediately by a
  **D015 epoch of 1703 s vs a steady ~875 s — nearly 2×** — then it recovered to 867 s once the load decayed
  (load avg peaked ~7.8). Not proven causal, but the timing is exact and the mechanism (page-cache eviction
  starving the dataloader) is plausible. **It is the 611 MHz throttled box: run assembly/scoring/eval on
  sie272 or the laptop while sie271 trains.** Cost here was ~15 min of wall-clock on a landing run.
- ⚠️ **`~/brats2025/.venv` is HIDDEN** — a `ls ~/brats2025/*` or `~/envs/*` glob **will not find it**, which is
  how the Synapse tooling got written off as "gone" (§0b). Check dotdirs before concluding something is missing.

---

## 8. DATA

**`Dataset007_BraTSMets2025`** — 1295 cases, 4-region: WT=[1,2,3] TC=[1,3] ET=[3] RC=[4], `regions_class_order=[2,1,3,4]`. Prevalence: NETC 54%, SNFH 89%, ET 96%, **RC 13%**.
**`Dataset011_BraTSMetsBackSplit`** — 6-region (adds NETC/SNFH aux). **Costs F1** (9769617). Symlinks D007's preprocessed data.
**`Dataset014_BraTSMetsSmallCC`** — §4. Symlinks D007's `.b2nd`; only the `.pkl` `class_locations` differ.

**⚠️ THE TRAINING ZIP HOLDS 1,296 CASES — ONLY 651 HAD EVER BEEN EXTRACTED.** Every cavity case sat in a **second site folder** (`UCSD - Training`) that nobody unpacked. **167 cases (12.9%) carry a cavity** — exactly the expected prevalence. Extracted 2026-07-14 to sie271 `~/brats2025/extracted_rc/`. **This is why the harness was blind to RC.** *(If something looks missing, check the zip before concluding it isn't there.)*

Ground truth for all 1295 is in `$nnUNet_preprocessed/Dataset007_BraTSMets2025/gt_segmentations/`.

---

## 9. 🔐 Synapse token (LOCAL ONLY — gitignored, do NOT commit)

```
<<REDACTED — token lives in ~/brats2025/syn.tok on sie271, NEVER commit>>
```

---

## 10. KEY SUBMISSIONS

**⚠️ 07-15, 07-16 AND 07-17 EACH SPENT BOTH SLOTS — six recent submissions.** `ours.py` lists all 44. The
07-17 pair: `9772250` (lo 0.20, extent bracketed) and `9772272` (D015, first F1 lift). What they settle:

### 10a. THE EXTENT AXIS IS CLOSED — bracketed 2026-07-17 (all mr in ONE snapshot, 689 subs)

| extent (lo) | sub | mean rank | DSC wt/tc/et ranks | F1_wt rank | verdict |
|---|---|---|---|---|---|
| 0.30 | 9771508 | 55.92 | 33/36/49 | 133 | old |
| **0.25** | **9771992** | **46.08** | 9/17/16 | 132 | 🏆 **optimum** |
| 0.20 | 9772250 | 46.50 | **6/15/10** | **149** | one step too far |

**Lowering `lo` (extent) buys DSC and costs F1_wt — and past 0.25 the trade goes negative.** 0.20 gained ~14
DSC-rank positions (packed field, cheap) but lost 17 F1_wt positions (sparse field, dear): net **+0.42 mr =
worse**. This is the §0 rank-density law made exact. **The extent sweep is DONE: 0.25 is the floor. Do not
submit lo<0.25 again.** The corollary matters more: **we cannot climb on DSC/NSD — we are 6–17th there and
~120–150th on F1. Only F1 moves us now.** That is the entire case for D015 (§4b) and the mimic work (§1c).

| sub | date | what | DSC wt/tc/et/rc | F1 wt/tc/et/rc |
|---|---|---|---|---|
| 🟢 **9772272** | 07-17 | `d015_dilateISO_standalone_erodek1_ccRC` (slot 2 today). **D015 dilate-trained model, standalone.** **F1 +.012/+.014/+.013 over the best — first F1 lift since hysteresis** — but DSC −.064/−.117/−.131 (unfused, expected). **NOT a submission; a FUSION DONOR** (§4b, §6 #2). Next slot = mask-union its new detections into the 3-way | .661/.649/.615/.630 | **.388/.492/.489**/.167 |
| **9771992** | 07-16 | 🏆 **CURRENT TEAM BEST** (mr 46.08 @ 689-snap). `slot2_3way_lo025hi070_extent` — 3-way fusion, hysteresis **lo 0.30 → 0.25**. **Extent paid: DSC +.005/+.005/+.009 for −.001 F1.** Optimum (§10a) | **.725/.766/.746/.630** | .376/.478/.476/.167 |
| 9772250 | 07-17 | `3way_hyst_lo020hi070_extentprobe` (slot 1 today). **lo 0.25 → 0.20.** DSC +.003/+.002/+.002 but **F1_wt −.003 → mr 46.50, slightly WORSE.** Brackets the extent optimum at 0.25 (§10a) | .728/.768/.748/.630 | .373/.478/.476/.167 |
| 9771991 | 07-16 | `slot1_3way_lo030hi070_RCmean_ccloss_fg` — the **RC ensemble** (mean of CC-loss + FgOversample). **RC .630 → .628: it did NOT help.** §5 called it unreliable and was right; §6 #4 (predict FgOversample to unlock it) is now **answered — drop it** | .725/.761/.737/.628 | .376/.479/.477/.167 |
| 9771836 | 07-15 | `3way_hyst_lo030hi075_probe` — seed **hi 0.70 → 0.75**. DSC flat, F1 −.011. **hi is already optimal at 0.70** | .724/.761/.737/.630 | .365/.469/.468/.167 |
| 9771835 | 07-15 | `3way_hyst_lo030hi065_probe` — seed **hi 0.70 → 0.65**. **F1 +.042/+.039/+.043, DSC −.017.** The §5 DSC↔F1 frontier again, from the seed side | .703/.744/.722/.630 | **.418/.518/.520**/.167 |

**READ: `lo` (extent) and `hi` (seed) are NOT the same knob.** Lowering **hi** rides the frontier — buys F1,
pays DSC, nets worse (9771835). Lowering **lo** was **free** — DSC up, F1 flat (9771992). The frontier §5
says not to fight is the **seed** axis. **The extent axis was never swept below 0.25.**

| sub | what | DSC wt/tc/et/rc | F1 wt/tc/et | mr |
|---|---|---|---|---|
| 9771576 | ex-best. = 9771508 with RC-fix removed. A **wash** (mr −0.2); the harness got the sign wrong — see §5 | .720/.761/.737/.614 | .376/.479/.477 | 56.7 |
| **9771508** | **3-way fusion + hyst .30/.70.** First submission ever chosen by measurement, not intuition. Superseded by 9771992 (same recipe, lo .25) | **.720/.761/.737/.630** | .376/.479/.477 | 56.9 |
| 9771379 | hyst .30/.60 on mean(topk,completion) | .718/.759/.740/.630 | .366/.469/.476 | 57.9 |
| 9771380 | hyst over the **CC-loss base**. **F1 = rank 3 of 1,422** in the whole competition. Sunk purely by FPs | .670/.715/.691/.630 | **.429/.530/.543** | 135 |
| 9771150 | **hysteresis introduced** — the single biggest lever ever found. #12 → #7 | .711/.753/.729/.630 | .365/.468/.475 | 75 |
| 9771002 | RC grafted from CC-loss (.581 → .630) | .679/.719/.696/.630 | .370/.477/.484 | 139 |
| 9770654 | ensemble TopK ⊕ completion @ step 0.25 | .678/.718/.696/.581 | .370/.477/.484 | 158 |
| 9769908 | **ensembling introduced** — super-additive | .666/.708/.684/.556 | .368/.439/.436 | 226 |
| 9768815 | baseline nnU-Net ResEnc-L | .560/.622/.593/.317 | .379/.469/.463 | 380 |

**Full log + ablation + step ladder: `~/CAI4CAI/brats2026final.html`.**

### The path: 380 → ~57 _(same-snapshot mr; see the time-dependence warning in §0)_
`baseline → RC-fix → TopK-BCE → overlap .375 → **ENSEMBLING (−62)** → recall-repaired member → overlap .25 → completion member → RC graft → **HYSTERESIS (−64)** → lower seed → **3-WAY FUSION**`

**The two biggest wins — ensembling and hysteresis — are both INFERENCE-TIME.** Neither needed a new model. Meanwhile the most expensive things we built (tumour specialist, cavity specialist, LoG hints, MedNeXt) contributed **nothing**.

---

## 11. HOW HYSTERESIS WORKS (the biggest lever — understand it)

```python
def hyst(p, lo, hi):                       # lo = 0.30, hi = 0.70
    lab, n = cc_label(p > lo)              # components of the GENEROUS mask
    cmax = np.zeros(n + 1, np.float32)
    np.maximum.at(cmax, lab.ravel(), p.ravel())   # PEAK probability inside each
    keep = cmax > hi                       # keep it iff that peak clears HI
    keep[0] = False
    return keep[lab]                       # paint the WHOLE component back
```

**`hi` decides WHETHER a lesion exists** (one confident voxel is enough). **`lo` decides HOW FAR it extends** (true tumour margin lives between 0.30 and 0.60). No single threshold does both: cut at 0.30 and you keep every speck; cut at 0.70 and you amputate every real lesion's edge.

**Why it is NOT the component filtering that failed four times:** those filtered on **size** or **mean** confidence — both correlated with *being a small real lesion*. Hysteresis filters on **peak** and **never looks at size**. A 30-voxel met with one voxel at 0.9 survives intact; a large diffuse blob that never reaches 0.7 anywhere dies. Applied per region; **RC excluded**.

---

## 12. 📝 THE SHORT PAPER — MANDATORY, DUE 2026-07-30 _(added 07-24)_

**Source: challenge wiki `syn74274097`, page 639582 "Submission Instructions" + 639585 "Challenge Rules". Verified 07-24.**

> ❗️ **Mandatory.** *"Failure to provide a short paper will disqualify your Docker submission from evaluation
> and final ranking."* + *"We will only run Docker submissions linked to a short paper."*

**Where:** OpenReview → `https://openreview.net/group?id=MICCAI.org/2026/Challenge`
(the wiki still carries a commented-out CMT link from 2025 — **OpenReview is the live one**).
Proceedings are published by **Springer LNCS**.

### Format — hard requirements
- **8–10 pages excluding references.** **Springer LNCS** template (`bit.ly/2TEcZNF`).
- Mandatory sections, in this order: **Abstract (⚠️ no citations allowed)** · **Keywords** · Introduction ·
  Methods · Results · Discussion · Acknowledgements (optional) · References.
- **Citations do not count toward the page limit.**
- **You must state our Synapse team name — `3495063` — in the submission form** so the paper links to our
  Docker. Get this wrong and the container is treated as paperless.

### Content — what it must report
1. **Source code: a GitHub link.** ⚠️ We do not currently have a public repo for this pipeline. **Creating one
   is on the critical path** — it is a named requirement, not a nicety.
2. **Method description.**
3. **Results on training AND validation data.** _(Our validation numbers are the leaderboard rows in §10 /
   `ours.py`; training-side numbers come from the local harness, §2 — mind its limits in §2/§5 when quoting them.)_

**The camera-ready is the identical paper plus the testing results.** ⇒ **Nothing in this paper waits on the
test set. It can be written in full today from what we already have.**

### Mandatory citations for Task 1 (BraTS-MET)
- **Flagship BraTS benchmark** — U. Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark…*, arXiv:2107.02314.
- **BraTS-MET 2023** — arXiv:2306.00838 (DOI 10.48550/arXiv.2306.00838). _The 2024 MET citation is listed
  "Coming soon" on the wiki — re-check page 639585 before camera-ready._
- **MedPerf** — A. Karargyris, R. Umeton, M.J. Sheller, et al., *"Federated benchmarking of medical artificial
  intelligence with MedPerf"*, **Nature Machine Intelligence 5:799–810 (2023)**, DOI 10.1038/s42256-023-00652-2
  — required for "any dataset and/or MedPerf client".
- Participants must also cite the challenge's own manuscript once released (organisers notify by email).

### Required acknowledgement string (verbatim)
> "Data used in this publication were obtained as part of the Challenge project through Synapse ID (syn74274097)."

### Obligations that follow submission
- **Peer review duty:** after submissions close we are invited to lightly review **2–3 papers**, answering
  (i) is anything missing in the methodological description? (ii) is anything missing the authors could fix
  within a week?
- **Withdrawal:** email `BraTSChallengeOrganizers@synapse.org` to pull the paper — **but then we are also
  dropped as co-authors on the BraTS journal manuscript** (arXiv:2107.02314 lineage).
- **Top-ranked methods MUST have a published article (preprint counts — arXiv/Zenodo) *before* the results
  announcement.** The short paper can serve as that article.
- **Publication embargo:** we may publish our own method's results and our own team ranking, but **not**
  overall challenge results or any analysis of them until the organisers' overview paper is out.
- **Poorly-performing methods may opt out** of the meta-analysis by name after MICCAI (methods still reported,
  anonymised).
- **Private data:** we use none — but note that if we ever did, the rules require it be declared in the
  manuscript **and** an additional MLCube submitted.
- **Licensing:** organisers are irrevocably permitted to publish our submitted container under **Apache 2.0**
  unless we state another licence.

---

## 13. 🐳 THE DOCKER CONTAINER — DUE 2026-07-30 _(added 07-24)_

**Source: challenge wiki `syn74274097`, page 639582. Verified 07-24.** Standard queues opened **Jul 3**;
Early queues (Jun 17 / Jul 2 deadline) are **not** our track.

> ❗️ *"Please confirm your short paper is submitted on OpenReview and your Synapse team name in that system is
> correct. We will only run Docker submissions linked to a short paper."*

### Compute envelope — Tasks 1–3 (segmentation)
| resource | limit |
|---|---|
| GPU | **NVIDIA A10G, 24 GB VRAM** |
| CPU / storage | 16 vCPUs · 200 GB |
| CUDA | **max 13.0** |
| `--memory` | **48 GiB** |
| `--shm-size` | **16 GiB** |
| **inference time** | **12 hours TOTAL for the whole hidden test set** |

🔴 **THE 12-HOUR TOTAL IS THE RISK AND IT IS UNMEASURED.** Our best submission is a **3-way fusion**
(`9771992`). At ~179 test cases that is ~4 min/case for three full ResEncL predictions + hysteresis decode.
Probably survivable at `step_size 0.25` — **but nobody has timed it end-to-end on A10G-class hardware, and our
boxes are GB10s, not A10Gs.** ⚠️ Note the A10G is **24 GB** and considerably slower than a GB10; do not
extrapolate our timings directly. **Measure this first.** It also permanently kills any finer step size in
the container (which is moot anyway — step 0.125 was refuted, 07-24 delta §7).

### Runtime rules — each of these invalidates the submission if broken
- **Zero network access.** Every dependency baked into the Dockerfile — **no runtime `pip install`**, weights
  inside the image.
- **`/input` is mounted read-only.** *"Any attempt by your model to write to or modify the `/input` directory
  will crash your run, thus invalidating your final submission."* ⚠️ nnU-Net likes to write next to its inputs —
  check this explicitly.
- **`/output` must be FLAT.** *"Do NOT create sub-folders; doing so will invalidate your final submission."*
- **Filenames** end with the 5-digit case ID + 3-digit timepoint, e.g. `BraTS-MET-12345-100.nii.gz`
  (or `12345-100.nii.gz`). One `.nii.gz` per case folder found in `/input`.
- **Spatial match is mandatory:** dimensions, voxel spacing, origin and orientation identical to the source
  image. *"This is especially important for the Metastasis task, as this dataset is in its native space."*
  Discrepancies ⇒ invalidation. (Same constraint our `verify` gate already enforces on the zips — reuse it.)

### ⚠️ Organisers do NOT check your container until after the queues close
*"Containers that fail to run or produce invalid outputs will not be evaluated."* **There is no second chance
and no error message.** Test locally with exactly their command:

```bash
docker run --rm --network none --gpus=all \
  --volume /PATH/TO/INPUT:/input:ro \
  --volume /PATH/TO/OUTPUT:/output:rw \
  --memory=48G --shm-size=16G \
  docker.synapse.org/PROJECT_ID/IMAGE_NAME:TAG
```
_(Use `--runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=0` instead of `--gpus=all` on the nvidia-runtime setup.)_
The 179-case validation set is the natural local `/input`.

**Push target:** `docker.synapse.org/PROJECT_ID/IMAGE_NAME:TAG`, then submit via the **Task 1 → Submission**
tab (`https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097#Submission`).
⚠️ Submission widgets are disabled unless registration **and** team membership are complete.

---

## 14. 🗓️ REAL TIMELINE _(added 07-24 — supersedes every date in §0 and §6)_

| date | what | note |
|---|---|---|
| ~~Jul 23~~ | ~~final deadline~~ | ❌ **struck out on the wiki, replaced by Jul 30** |
| **Jul 30, 23:59 UTC** | 🚨 **FINAL DEADLINE** | **Docker + short paper + signed copyright form.** ALL queues close, **including the Validation Leaderboard** |
| Aug 6 | Reviews available, initial decision | Accept / Revision / Reject |
| Aug 13 | Revised-paper deadline | for "Revision" decisions only |
| Aug 24 | Final acceptance + invitation to present at MICCAI 2026 | |
| Aug 28 | Top teams contacted for oral presentations | oral-vs-poster announced |
| Sep 27 – Oct 1 | MICCAI 2026 satellite events | |

**We are Track B (Standard).** Track A (Early) closed Jul 2 and we did not use it — which is fortunate, since
using it would have *barred* us from submitting on Jul 30.
