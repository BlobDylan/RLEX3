# Project Roadmap — Deep RL in MiniGrid (2026 B)

Living checklist for the final project. Update the **Current step** marker as we finish each item.

---

## Current step

> **Step 3 — Make DQN solve `SimpleRoomEnv` (sanity check)**
>
> Status: **in progress — freeze obs pipeline; next = model / learning changes**
>
> Locked preprocessing (for now):
> `tile=12 → wall-crop → tile-inset(0.75) → RGB → resize 64` + shaping `+50/−0.1` + actions `(0,1,2)`
>
> Exp3–5: train can hit high `succ20`, but greedy collapses (forward-walk or spin). RGB Exp5 ≈ same as gray.
>
> **Next model levers (priority):**
> 1. **Dueling DQN** (V + A streams) — often stabilizes action gaps
> 2. Stronger spatial head (less aggressive downsample / bigger feature map)
> 3. Learning tweaks: slightly higher `eps_end`, more steps, or n-step returns
>
> Do not chase more preprocessing until greedy eval is clearly better.

---

## Progress overview

| Step | Topic | Status |
|------|--------|--------|
| 0 | Setup & template | done |
| 1 | Understand envs + explorer | done |
| 2 | Baseline wrappers + `BaseAlgorithm` + DQN scaffold | done |
| **3** | **DQN solves `SimpleRoomEnv`** | **← now** |
| 4 | Observation pipeline (proper preprocessing) | pending |
| 5 | Training graphs & logging | pending |
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

## Step 3 — DQN must solve `SimpleRoomEnv` ← **YOU ARE HERE**
**Status:** in progress

**Goal (pass/fail):** greedy policy reaches the green goal most of the time. If it cannot, the bug is in preprocessing / net / hypers / loop — not in `ComplexEnv`.

### 3.1 Diagnose the current failure
- [x] Confirm reward signal: success logged via `terminated` (`episode_success` / `succ20`)
- [ ] Confirm obs pipeline: shape `(40,40,1)`, values in `[0,255]`, CNN sees NCHW correctly
- [x] Confirm actions: `{left, right, forward}` is enough for empty room
- [ ] Check learning actually happens: loss decreases, Q-values grow, ε decays
- [x] Check train budget: notebook now targets `50_000` steps (was `5_000`)

### 3.2 Likely fixes to try (in order)
1. Longer training (`50k`–`200k` steps) with logging of **success rate**
2. [x] Stronger / clearer reward for SimpleRoom — `SimpleRoomShapingWrapper` (`+50` goal, `-0.1`/step)
3. Hyperparameters: higher `lr`, larger early exploration, smaller `learning_starts`, more frequent target updates
4. Network / input: normalize, maybe keep RGB or resize differently if grayscale loses too much
5. Bugs: done masking (`terminated` vs `truncated`), target copy, buffer dtypes

### 3.3 Exit criteria
- [ ] Training curves: return ↑, length ↓, success rate → high
- [ ] Greedy eval over ≥10 seeds: high success rate, return near `+1`
- [ ] Short video of a successful episode saved under `videos/`
- [ ] Log the run in `EXPERIMENTS.md` + save plots under `graphs/`

**Do not start ComplexEnv agents until exit criteria are met.**

---

## Step 4 — Observation preprocessing (final pipeline)
**Status:** pending

Hard rule: **pixels only** into the network. Getters / positions = shaping & EDA only.

For each env, document and implement:
- [ ] Resolution / resize
- [ ] Color vs grayscale
- [ ] Channel order + normalization
- [ ] Frame stack (yes/no + why)
- [ ] Final tensor shape fed to the net
- [ ] Action subset kept per task (state explicitly)

Prefer implementing this as real `gym.ObservationWrapper`s (and reuse for all algorithms).

---

## Step 5 — Training metrics & graphs
**Status:** pending

Assignment requires (x-axis = training episode; rolling averages OK):
- [ ] Reward / return per episode
- [ ] Steps per episode
- [ ] Success rate per episode
- [ ] **Cumulative env steps vs episode** (common budget axis)

Also:
- [ ] Helpers to save figures into `graphs/`
- [ ] Log hypers + seeds in `EXPERIMENTS.md`

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
- [ ] Successful SimpleRoom video (after Step 3)
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
