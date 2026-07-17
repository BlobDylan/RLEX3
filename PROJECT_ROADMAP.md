# Project Roadmap — Deep RL in MiniGrid (2026 B)

Living checklist for the final project. Update the **Current step** marker as we finish each item.

---

## Current step

> **Step 5 — Training metrics & graphs**
>
> Status: **in progress** — helpers + notebook cell added; run the Step 5 cell to write PNGs
>
> **Next:** execute the Step 5 notebook cell (uses `dqn_gray_history` / `dqn_best_history`
> or a JSON under `graphs/`), confirm the four plots look right, then Step 6 ComplexEnv.

---

## Progress overview

| Step | Topic | Status |
|------|--------|--------|
| 0 | Setup & template | done |
| 1 | Understand envs + explorer | done |
| 2 | Baseline wrappers + `BaseAlgorithm` + DQN scaffold | done |
| 3 | DQN solves `SimpleRoomEnv` | **done** |
| 4 | Observation pipeline (document / lock) | **done** (SimpleRoom; gray after Exp7) |
| **5** | **Training graphs & logging** | **← now** |
| 6 | DQN on `ComplexEnv` (+ shaping) | pending |
| 7 | Policy-based method (e.g. REINFORCE) | pending |
| 8 | Actor-critic method (e.g. PPO / A2C) | pending |
| 9 | Full training results & ComplexEnv stage progress | pending |
| 10 | Inference comparison table / charts | pending |
| 11 | Discussion + best settings cell | pending |
| 12 | Videos (mid-training + converged) | partial |
| 13 | Report + submission files | pending |

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

*Report later — do not block Step 3 on polished write-up.*

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
**Status:** done (SimpleRoom); ComplexEnv when Step 6 starts

Hard rule: **pixels only** into the network. Getters / positions = shaping & EDA only.

**Frozen SimpleRoom stack (after Exp7):**
`tile_size=12` → crop outer walls → tile inset `0.75` → **grayscale** → resize `64×64×1`
(+ `SimpleRoomShapingWrapper`, actions `{left, right, forward}`)

Exp4 gray-spin was bad hypers, not grayscale (Exp7: 90% greedy @ 20k, no spin).
RGB remains a valid alternative; prefer **RGB on ComplexEnv** (color-coded objects).

For each env, document and implement:
- [x] Resolution / resize — `64×64`
- [x] Color vs grayscale — **grayscale for SimpleRoom** (Exp7); RGB for ComplexEnv later
- [x] Channel order + normalization — HWC uint8 → NCHW float in DQN
- [x] Frame stack — no (static room)
- [x] Final tensor shape — `(64, 64, 1)` → CNN `(B, 1, 64, 64)`
- [x] Action subset — `{0,1,2}` left/right/forward
- [x] Write this stack clearly in the notebook
- [ ] Adapt / document ComplexEnv variant when starting Step 6

Prefer implementing this as real `gym.ObservationWrapper`s (and reuse for all algorithms).

---

## Step 5 — Training metrics & graphs
**Status:** in progress (helpers ready; run notebook cell)

Assignment requires (x-axis = training episode; rolling averages OK):
- [x] Reward / return per episode — `plot_training_history` → `*_return.png`
- [x] Steps per episode — `*_length.png`
- [x] Success rate per episode — `*_success.png`
- [x] **Cumulative env steps vs episode** — `*_cum_steps.png` (+ overview panel)

Also:
- [x] Helpers to save figures into `graphs/` — `algorithms/plotting.py`
- [x] Persist history JSON for kernel restarts
- [ ] Confirm plots from Exp7 / Exp6b history in the notebook
- [ ] Log hypers + seeds already in `EXPERIMENTS.md` (keep updated)

---

## Step 6 — DQN on `ComplexEnv`
**Status:** pending

- [ ] Action subset for the full mission (drop unused `done`; justify kept set)
- [ ] Legal reward shaping wrapper (event-based OK; **no distance/geometry**)
- [ ] Train DQN; expect partial progress
- [ ] Stage metrics: key → door → water → lava → goal (even if goal rarely reached)
- [ ] Mid-training + post-training videos
- [ ] Graphs for ComplexEnv (including cumulative steps)

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
- [ ] ComplexEnv progress videos

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

## Immediate next session plan (Step 3)

1. Add success-rate logging to DQN training history.
2. Run a longer SimpleRoom experiment and plot return / success / length.
3. Patch whatever is broken (hypers, reward visibility, bugs).
4. Only when greedy success is clearly high → mark Step 3 done and start Step 4/5 polish, then Step 6.
