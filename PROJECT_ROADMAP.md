# Project Roadmap — Deep RL in MiniGrid (2026 B)

Living checklist for the final project. Update the **Current step** marker as we finish each item.

---

## Current step

> **Step 6c done / 6d in progress — ComplexEnv from-scratch DQN**
>
> Pure from-scratch Double-DQN on ComplexEnv (single `(64,64,3)` RGB frame,
> monotonic pull-forward shaping, n-step, `n_envs=8`, `width_mult=2`, MPS). Runs +
> graphs + periodic rollout videos captured. Bottleneck isolated to the
> **water→lava→goal ferry** plus a **greedy-vs-exploratory gap** (permanent ε floor
> left the greedy policy trap-bound).
>
> **Next:** (A) low-ε anneal (→0.05 / 700k) + 900k budget alone to fix the greedy
> traps; then (A)+**scale-robust prioritized replay** (max-tree) to consolidate the
> ferry. Full run-by-run log in **Step 6d** below.

---

## Progress overview

| Step   | Topic                                                    | Status                                 |
| ------ | -------------------------------------------------------- | -------------------------------------- |
| 0      | Setup & template                                         | done                                   |
| 1      | Understand envs + explorer                               | done                                   |
| 2      | Baseline wrappers + `BaseAlgorithm` + DQN scaffold       | done                                   |
| 3      | DQN solves `SimpleRoomEnv`                               | **done**                               |
| 4      | Observation pipeline (document / lock)                   | **done** (SimpleRoom; gray after Exp7) |
| 5      | Training graphs & logging                                | **done**                               |
| 6a     | ComplexEnv: stack + event shaping                        | **done**                               |
| 6b     | ComplexEnv: stage metrics + short DQN smoke              | **done**                               |
| **6c** | **ComplexEnv: first real DQN train + graphs/videos**     | **done**                               |
| **6d** | **ComplexEnv: unstick (shaping + exploration + replay)** | **← now**                              |
| 7      | Policy-based method (e.g. REINFORCE)                     | pending                                |
| 8      | Actor-critic method (e.g. PPO / A2C)                     | pending                                |
| 9      | Full training results & ComplexEnv stage progress        | pending                                |
| 10     | Inference comparison table / charts                      | pending                                |
| 11     | Discussion + best settings cell                          | pending                                |
| 12     | Videos (mid-training + converged)                        | partial                                |
| 13     | Report + submission files                                | pending                                |

---

## Step 0 — Setup & template

**Status:** done

- [x] Poetry / venv, GPU (`mps`), notebook kernel
- [x] Template notebook + `env_explorer.py`
- [x] `EXPERIMENTS.md`, `graphs/`, `videos/`

---

## Step 1 — Understand the environments

**Status:** done (refine later in the report)

**What this step is for:** characterize both MDPs before coding more.

- [x] Drive `SimpleRoomEnv` / `ComplexEnv` (explorer / random rollouts)
- [ ] Write MDP notes for the report (episodic, pixel obs, observability, state-space size estimate)
- [ ] Note differences that matter: horizon, sparsity, lava = lethal, recoverable vs non-recoverable mistakes

_Report later — do not block Step 3 on polished write-up._

---

## Step 2 — Scaffold: wrappers + algorithm base + DQN

**Status:** done

- [x] Example wrappers: `GrayscaleWrapper`, `EventShapingWrapper`, `ActionSubsetWrapper`
- [x] `algorithms/base.py` — `BaseAlgorithm`
- [x] `algorithms/dqn.py` — replay, CNN Q-net, target net, ε-greedy, train/eval/save
- [x] Notebook cells under **Your Code Below**

---

## Step 3 — DQN must solve `SimpleRoomEnv`

**Status:** done

**Goal (pass/fail):** greedy policy reaches the green goal most of the time.

### 3.3 Exit criteria

- [x] Training curves: return ↑, length ↓, success rate → **100% `succ20` sustained**
- [x] Greedy eval strong in Exp6 (@20k: **90%**); Exp6b train confirms solve
- [x] Success video path in notebook (`dqn_simple_room_best.mp4`)
- [x] Logged in `EXPERIMENTS.md` (Exp6 / Exp6b)

**Baseline to keep:** grayscale `64×64×1` pipeline (Exp7) + Exp6 winner hypers.
(RGB also works @ 90% — Exp6; kept as ComplexEnv candidate.)

---

## Step 4 — Observation preprocessing (final pipeline)

**Status:** done (SimpleRoom); ComplexEnv variant in **6a**

Hard rule: **pixels only** into the network. Getters / positions = shaping & EDA only.

**Frozen SimpleRoom stack (after Exp7):**
`tile_size=12` → crop outer walls → tile inset `0.75` → **grayscale** → resize `64×64×1`
(+ `SimpleRoomShapingWrapper`, actions `{left, right, forward}`)

Exp4 gray-spin was bad hypers, not grayscale (Exp7: 90% greedy @ 20k, no spin).
RGB remains a valid alternative; prefer **RGB on ComplexEnv** (color-coded objects).

For each env, document and implement:

- [x] Resolution / resize — `64×64`
- [x] Color vs grayscale — **grayscale for SimpleRoom** (Exp7); **grayscale for ComplexEnv 6a** (user choice)
- [x] Channel order + normalization — HWC uint8 → NCHW float in DQN
- [x] Frame stack — no (static room)
- [x] Final tensor shape — `(64, 64, 1)` → CNN `(B, 1, 64, 64)`
- [x] Action subset — `{0,1,2}` left/right/forward (SimpleRoom); ComplexEnv `(0..5)` in 6a
- [x] Write this stack clearly in the notebook
- [ ] Adapt / document ComplexEnv variant (**6a** — gray stack in notebook)

Prefer implementing this as real `gym.ObservationWrapper`s (and reuse for all algorithms).

---

## Step 5 — Training metrics & graphs

**Status:** done

Assignment requires (x-axis = training episode; rolling averages OK):

- [x] Reward / return per episode — `plot_training_history` → `*_return.png`
- [x] Steps per episode — `*_length.png`
- [x] Success rate per episode — `*_success.png`
- [x] **Cumulative env steps vs episode** — `*_cum_steps.png` (+ overview panel)

Also:

- [x] Helpers to save figures into `graphs/` — `utils/plotting.py`
- [x] Persist history JSON for kernel restarts
- [x] Confirmed plots from Exp7 history (`graphs/dqn_gray_history_*.png`)
- [x] Hypers + seeds logged in `EXPERIMENTS.md` (keep updated)

---

## Step 6 — DQN on `ComplexEnv` (phased)

Hard mission: key → door → water → lava → goal. Expect **partial progress** at first.
Do **not** start a long train until **6a** (and ideally **6b**) exit criteria pass.

### 6a — Env stack & event shaping

**Status:** done

- [x] Action subset `(0..5)`; grayscale `64×64×1`; first-time anti-farm shaping
- [x] Locked magnitudes: `goal_scale=50`, `step_penalty=0.1`, keep `enter_right_room=+5`
- [x] `max_steps=200` via factory (env default is 400; assignment allows override)
- [x] Sanity random rollouts: events fire (key often; door/right rare)
- [x] Documented in `EXPERIMENTS.md`

### 6b — Stage metrics + short DQN smoke

**Status:** done

- [x] Log stage reach-rates in training history / printouts (`stage_key` … `stage_goal`)
- [x] Prefer `info["success"]` over raw `terminated` (lava death ≠ win)
- [x] Throughput: `train_freq` / sparse loss sync (Exp10); **Exp10b** Mac defaults (`n_envs=1`, `freq=8`)
- [x] Short DQN smoke (~10k) — stage logs OK; Exp10b re-run **~35s** on MPS

**Exit:** metrics wired; smoke does not crash; early stages move (even if goal = 0%). ✓

### 6c — First real train + graphs / videos

**Status:** done

- [x] Longer budget (from-scratch runs at 500k–900k steps)
- [x] Graphs via Step 5 helpers (return / length / success / cum steps + stage curves)
- [x] Mid-training + post-training videos (periodic 6-episode rollouts every 100k)
- [x] Identify bottleneck stage for the report (**water→lava→goal ferry**)

**Exit:** logged runs + plots + videos; “how far does DQN get?” answered — reliably
key→door→right→water; ferry (lava/goal) is the open frontier. ✓

### 6d — Unstick (in progress)

**Status:** in progress — pure from-scratch DQN; iterating on shaping / exploration / replay.

**Corrected diagnosis (supersedes the earlier 2026-07-18 notes):**

1. **Inventory IS observable from a single RGB frame.** An earlier note claimed the
   carried object is invisible (frame diff = 0) and that a memoryless CNN therefore
   cannot represent the water→lava policy. Experiment **disproved** this: a single
   `(64,64,3)` RGB frame is sufficient, and **frame-stacking HURT** (≈57% → 10% on the
   door skill). We use one RGB frame; no recurrence needed.
2. **Camping local optima are the core failure mode**, not observability. Whatever the
   last reliably-reached milestone is, the greedy policy learns to reach it and then
   idle to timeout — banked reward + a small step penalty beats the long/risky next
   stage. Countered stage-by-stage with a **monotonically increasing “pull-forward”
   reward staircase** (each milestone strictly worth more than the last) so the value
   gradient always points forward.
3. **Reward-hacking guards.** key / door / right / key-drop bonuses are first-time
   latched; key-drop pays **once and only with the door already open** (no pick/drop
   farm); water paid once per original spawn tile; lava-extinguish deletes the tile.

**Experiment journey (ComplexEnv, from-scratch Double-DQN):**

| Run      | Change                                                                             | Result                                                                                                                                           |
| -------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Base     | single RGB frame, event shaping, `reward_scale`, n-step, Double-DQN, MPS           | key/door learned; door→right stalled                                                                                                             |
| Capacity | `width_mult` 1→2 (0.5M→2M params) + `n_envs=8`                                     | **cracked door→right** — a capacity+exploration wall, not shaping                                                                                |
| N+1      | pull-forward rewards (water 20, lava 25, goal 120, key_drop 15); `feat_hw` 5×5→8×8 | water 55–80%; first goal touches; camping recurred **post-water** (~+50 plateau)                                                                 |
| N+2      | ε-floor 0.20 / decay 400k; `max_steps` 300→200; `n_step` 3→5; 600k                 | training `succ20` spiked to 25% in the tail — but **final greedy eval = 0%** (key 72, water 4, lava/goal 0)                                      |
| diag     | fixed-ε sweep on the N+2 model                                                     | greedy ≪ exploratory: early chain **trap-bound** (tiny ε fixes key 72→92); **ferry ~0% at every ε** (never learned)                              |
| N+3      | (A) ε→0.05 / 700k anneal + 900k; (B) prioritized replay                            | A+B **regressed** (PER + aggressive `reward_scale` → recency-biased replay). Split: run A alone; PER **fixed** with a scale-robust **max-tree**. |

**Key structural findings (for the report):**

- **Pull-forward shaping** beats camping, but the camp relocates to the next frontier.
- **Greedy-vs-exploratory gap:** a permanent ε floor leaves the greedy policy
  trap-bound; the greedy trajectory must be optimised by **annealing ε low** at the end.
- **The water→lava→goal ferry is a rare-event learning wall** — reachable under
  exploration but never consolidated into the greedy policy; current lever is
  scale-robust prioritized replay (up-sample the rare high-TD ferry transitions).

- [x] Retune shaping magnitudes / step penalty / `max_steps` (pull-forward staircase; timeout == death, no suicide)
- [x] Obs / arch tweaks (single RGB confirmed; `feat_hw` grid-aligned 8×8; 2M–4.6M params)
- [x] Exploration schedule (sustained ε for discovery; low-ε anneal for greedy consolidation)
- [x] Prioritized replay (scale-robust max-tree) to up-sample rare ferry transitions
- [ ] Land a >0% **greedy** eval on the full chain, or write up the justified partial (key→door→right→water)

**Exit:** better stage progress **or** a justified partial-success write-up.

---

## Step 7 — Policy-based algorithm (e.g. REINFORCE)

**Status:** pending

- [ ] Implement from scratch (PyTorch nets only; no SB3 / RLlib / …)
- [ ] Run on **both** envs
- [ ] Same graphs + inference protocol as DQN
- [ ] Exploration: entropy / stochastic policy — document it

---

## Step 8 — Actor-critic algorithm (e.g. PPO or A2C)

**Status:** pending

- [ ] Implement from scratch
- [ ] Run on **both** envs
- [ ] Same graphs + inference + ComplexEnv stage progress
- [ ] Discuss sample efficiency / stability vs DQN & REINFORCE

---

## Step 9 — Training results & analysis

**Status:** pending

- [ ] All required plots for every algorithm × env
- [ ] ComplexEnv: stage-progress analysis (not just success/fail)
- [ ] Efficiency note: fewer total env steps is a small competition edge

---

## Step 10 — Inference results

**Status:** pending

Greedy eval on fresh seeds:

- [ ] Avg return, avg steps, success rate — all algos × both envs
- [ ] ComplexEnv stage depth
- [ ] Clear comparison table and/or grouped bar chart

---

## Step 11 — Discussion & best settings

**Status:** pending

- [ ] Strengths / weaknesses per family × env
- [ ] What shaping helped vs unintended incentives
- [ ] Dedicated notebook cell: **best training & inference settings**

---

## Step 12 — Videos

**Status:** partial

Rules: short clips **partway through training** and **after convergence**, in the notebook.

- [x] Smoke video for early DQN on SimpleRoom (failing)
- [x] Successful SimpleRoom video (`videos/dqn_simple_room_best.mp4`)
- [ ] Mid-training + converged videos for each algo worth showing
- [ ] ComplexEnv progress videos (Step **6c**)

---

## Step 13 — Report & submission

**Status:** pending

- [ ] `report_ID1_ID2.pdf` ≤ 12 pages, fully self-contained
- [ ] `details.txt` (Colab link + partner IDs)
- [ ] optional `explainer.txt`
- [ ] Notebook on Colab with **all outputs already present**
- [ ] Seeds + every hyperparameter documented
- [ ] Extension beyond baseline (needed for a perfect score — plan late)

---

## Hard constraints (do not violate)

1. Implement algorithms yourself — no SB3 / RLlib / Tianshou / CleanRL.
2. Network input = **pixels only**; no getters/positions as features.
3. No distance- or geometry-based rewards.
4. Env classes fixed except `max_steps`.
5. Own code below **Your Code Below**; wrappers for preprocess / shaping / actions.
6. Seed Python, NumPy, and the DL framework.

---

## Immediate next session plan (Step 6a)

1. Implement `ComplexShapingWrapper` (event bonuses + stage latches in `info`).
2. Build `make_complex_env()`: RGB obs pipeline + action subset + shaping.
3. Sanity-check events (explorer / scripted) — **no long DQN train**.
4. Log the design in `EXPERIMENTS.md`, then move to **6b**.
