# Report Memos — Deep RL in MiniGrid (2026 B)

**Purpose:** single source of truth for the final report. Everything we tried, the
results, what worked, what failed, and the interesting findings to present. Distil the
report from this file. Plots/figures live in [`graphs/`](graphs/); videos in
[`videos/`](videos/) and under each run's `videos/` dir.

> Scope note: this file **replaces** the old `EXPERIMENTS.md` (detailed run log) and
> `PROJECT_ROADMAP.md` (plan/status). It keeps the useful content of both — the
> experiment narrative for the report, plus the open TODO for finishing the project.

---

## 1. Project at a glance

Train deep RL agents **from pixels** on two image-based MiniGrid envs and compare three
from-scratch algorithms.

| Env             | Task                                    | Win condition                                                |
| --------------- | --------------------------------------- | ------------------------------------------------------------ |
| `SimpleRoomEnv` | empty 10×10 room (sanity check)         | step onto the green goal                                     |
| `ComplexEnv`    | key → locked door → water → lava → goal | ferry water to extinguish the lava ring, then reach the goal |

| Algorithm | Family       | Status                                                       |
| --------- | ------------ | ------------------------------------------------------------ |
| DQN       | value-based  | working; ComplexEnv **52% greedy** at 4M (stalled improving) |
| REINFORCE | policy-based | implemented; collapse issue diagnosed + hardened             |
| PPO       | actor-critic | implemented; greedy-gap diagnosed + LR-anneal fix            |

### Assignment constraints & grading levers (from the brief)

- **Pixels only.** Network sees the _rendered image_ (single `(64,64,3)` RGB frame). **No**
  underlying state / grid coords / object geometry. Reward shaping may only use env
  **events** (key picked, door opened, tile extinguished) — never geometric distances.
- **From scratch.** DQN / REINFORCE / PPO in raw PyTorch. **No SB3 / RLlib / Tianshou /
  CleanRL** — only `torch`, `numpy`, `gymnasium`, MiniGrid.
- **Envs are fixed** except the allowed **`max_steps`**. All difficulty tuning lives in
  wrappers (obs pipeline + event shaping), never the env.
- **Seed everything** (Python, NumPy, torch). Eval on **fresh/held-out seeds** (we use
  seed 9999, distinct from training).
- **Required plots per algorithm × env:** episode return, episode length, success rate,
  loss, and a **reward-vs-cumulative-env-steps** curve (sample-efficiency axis). All from
  `utils.plot_training_history`.
- **ComplexEnv stage-progress analysis**, not just success/fail — report the
  `key→door→right→water→lava→goal` cascade. **Partial success is explicitly valued.**
- **Inference comparison** — greedy (and small-ε) eval table across the three algorithms
  on both envs, fresh seeds.
- **Videos required** — mid-training and converged. We tile 20 greedy episodes into one
  grid mp4 (`utils.grid_rollout_video`) every 100k steps.
- **Report ≤ 12 pages, self-contained.** Distil from this file.
- **Extension for a perfect score** — beyond the three required algorithms. Candidate:
  the PER max-tree and within-episode ε-ramp ablations (already run).

---

## 2. Observation pipeline (final, locked)

Hard rule: **pixels only** into the network. Getters/positions are for shaping & EDA only.

**SimpleRoom (frozen after Exp7):**
`tile_size=12` → crop outer walls → tile inset `0.75` → **grayscale** → resize `64×64×1`
(+ `SimpleRoomShapingWrapper`, actions `{left, right, forward}`).

**ComplexEnv:** single **RGB** `64×64×3` frame (color matters — key/lava/water are
colour-coded), actions `(0..5)`, `max_steps=200` (env default 400; override allowed).

- HWC `uint8` → NCHW float `/255` **inside** the network.
- **Frame stack: NO.** A single RGB frame is sufficient; **frame-stacking HURT** (≈57% →
  10% on the door skill). Inventory _is_ readable from one frame — an earlier "carried
  object is invisible / needs recurrence" hypothesis was **disproved**.
- Shared CNN trunk (`algorithms/networks.py::CNNEncoder`, also in DQN's Q-net): 3 conv
  layers (stride 2,2,1) → optional extra convs → interpolate to `feat_hw=(8,8)` → FC 256.
  `feat_hw=8×8` is a clean 2× downsample of the 16×16 conv map, matching the 8×8 interior
  grid — preserves the per-cell adjacency the water→lava toggle depends on (grid-aligned
  beat the earlier 5×5).

---

## 3. Reward shaping design (ComplexEnv)

Event-based only (no geometry). Small anti-farm milestone bonuses guide exploration; the
**goal is dominant** so the agent can't bank a comfortable return by camping.

- **Monotonic pull-forward staircase:** each milestone strictly worth more than the last
  (`key < door < right < water < lava-extinguish < goal`) so the value gradient always
  points forward. Beats camping — but the camp **relocates to the next frontier**.
- **Anti-hack guards:** key / door / right / key-drop bonuses first-time latched; key-drop
  pays **once and only with the door already open**; water paid **per original spawn tile**;
  lava-extinguish **deletes** the tile.
- **Final tuned magnitudes (4M run):** `goal_scale=160`, `step_penalty=0.05`,
  `key_pickup=5`, `door_open=10`, `enter_right_room=15`, `key_drop=15`, `water_pickup=30`
  (per bucket), `lava_extinguish=50`, `lava_death=8`. `reward_scale=0.02` keeps Q ≈ O(1).

---

## 4. Results so far

### SimpleRoom (sanity check)

| Algo      | Greedy success  | Notes                                                        |
| --------- | --------------- | ------------------------------------------------------------ |
| DQN       | **~90–100%**    | Exp6/6b winner hypers; PASSED sanity check                   |
| REINFORCE | ~4% (collapsed) | entropy collapse; defaults hardened, re-run pending          |
| PPO       | **100%**        | **solved** — 100% greedy, converges ~23–30k steps (see §5.5) |

### ComplexEnv (hard task)

| Algo      | Greedy success      | Notes                                                              |
| --------- | ------------------- | ------------------------------------------------------------------ |
| DQN       | **52%** (goal) @ 4M | full chain; still improving at 4M — resumed for more steps         |
| REINFORCE | not yet run         | —                                                                  |
| PPO       | not yet run         | expected to beat DQN on the ferry (on-policy, no replay bootstrap) |

**4M greedy stage cascade (eval, 50 eps, seed 9999):** key 96% → door 94% → right 90% →
water 62% → lava 54% → **goal 52%**. Training `succ20` reached 55–85% in the tail
(`mean20` return ~200–285) and was **still trending up** at 4M — hence the resume. The
remaining drop-off is the water→lava→goal ferry (now the _only_ soft stage; key/door/right
are ~100%). This is the headline DQN result — a big jump over the earlier 10–15%, driven by
the **moderate-magnitude shaping + `n_step=5` + long low-ε consolidation tail**.

---

## 5. Experiment journey

### 5.1 SimpleRoom DQN — the greedy-collapse saga (Exp 1–7)

The recurring theme: **training explores its way to the goal, but the greedy policy
collapses** into a degenerate behavior. Fixed by hyperparameters, not architecture.

| Exp | Setup                                                 | Result / finding                                                                        |
| --- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1   | sparse smoke, 5k steps, gray 40×40                    | no goal — too short + sparse                                                            |
| 2   | + `SimpleRoomShapingWrapper(goal 50, step −0.1)`, 50k | train `succ20` 5–40%; **greedy eval 0%** (greedy collapses)                             |
| 3   | Double DQN, soft τ=0.005, smaller CNN                 | train 60–70%; greedy **always forward** `{0:1,1:0,2:99}` → eval 40%                     |
| 4   | + wall-crop + tile-inset 0.75 + resize64 (gray)       | train 70–80%; greedy **spin** `{0:47,1:47,2:6}` → eval 20%                              |
| 5   | Exp4 pipeline but RGB                                 | similar; no clear greedy improvement                                                    |
| 6   | random hyperparam search (8×20k)                      | **Trial 4: greedy 90%**, len≈19 (`lr≈2.8e-4, γ=0.95, batch=128, τ=0.001, eps_end=0.15`) |
| 6b  | full 50k retrain with Exp6 winner                     | **`succ20` 100% by ~ep220, stays 100%** — SimpleRoom **PASSED**                         |
| 7   | grayscale A/B with winner hypers                      | gray ≈ RGB (90% greedy, no spin) → **Exp4 spin was hypers, not grayscale**              |

**Takeaways for the report:** (a) the _stochastic_ training policy solving a task tells
you little about the _greedy_ policy; (b) forward-collapse / spin are the two failure
modes of a brittle deterministic policy; (c) a small random hyperparameter search was the
single most effective fix; (d) grayscale ≈ RGB here.

### 5.2 ComplexEnv DQN — throughput & the door bottleneck (Exp 10–13)

- **Exp 10 / 10b (throughput):** wall-clock was **update-bound**, not env-bound. Q-net is
  ~0.5M params (~2 MB) → low GPU RAM is normal. `train_freq=4` gave ~4–5× over
  `train_freq=1`; `loss.item()` host-sync every update was a hidden cost (fixed with
  `log_loss_every`). On **Mac MPS** prefer `n_envs=1` + high `train_freq`; on **CUDA**
  `n_envs=4..8` helps. `SyncVectorEnv` runs envs serially → don't over-update per vec step.
- **Exp 11 (long CUDA run, 132k):** `succ20=0%` throughout; key oscillates 20–70% (no
  trend), **door ~0%**. Diagnosis: **key is reachable by chance; the door is the hard
  bottleneck**; more of the same budget won't finish. Score trials by `stage_door` /
  `stage_right`, not goal alone.
- **Exp 12 (overnight sweep, 47 trials):** shaping/hparams alone mostly fail like Exp11 —
  **except `train_freq=2` ("frequent_updates")**, which finally moves the door.
  Winners: `E_deep_frequent` (door 46%, right 42%), `B_dqn_frequent_updates` (door 44% @
  80k). Greedy diagnose on winners still often **collapsed** → need anti-collapse.
- **Exp 13 (tile × width A/B):** `width_mult=2` (~1.9M) beat `width_mult=1` on late door
  (56% vs 38%); `tile=16` weak. Locked: `tile=12`, `width_mult=2`.

### 5.3 ComplexEnv DQN — the unstick journey (from-scratch Double-DQN)

Core failure mode = **camping local optima**: the greedy policy reaches the last reliable
milestone and idles to timeout (banked reward + small step penalty beats the long/risky
next stage). Countered with the pull-forward staircase + exploration/replay tooling.

| Run          | Change                                                                                    | Result                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Base         | single RGB, event shaping, `reward_scale`, n-step, Double-DQN, MPS                        | key/door learned; door→right stalled                                                                                             |
| Capacity     | `width_mult` 1→2 (0.5M→2M) + `n_envs=8`                                                   | **cracked door→right** — capacity+exploration wall, not shaping                                                                  |
| N+1          | pull-forward (water 20, lava 25, goal 120, key_drop 15); `feat_hw` 5×5→8×8                | water 55–80%; first goal touches; camping recurred **post-water** (~+50 plateau)                                                 |
| N+2          | ε-floor 0.20 / decay 400k; `max_steps` 300→200; `n_step` 3→5; 600k                        | train `succ20` 25% in tail — but **greedy eval 0%** (key 72, water 4, lava/goal 0)                                               |
| diag         | fixed-ε sweep on N+2 model                                                                | greedy ≪ exploratory: early chain **trap-bound** (tiny ε fixes key 72→92); **ferry ~0% at every ε**                              |
| N+3          | (A) ε→0.05 / 700k anneal; (B) prioritized replay                                          | A+B **regressed** (PER + aggressive `reward_scale` → recency bias). PER **fixed** with a scale-robust **max-tree**               |
| tooling      | max-tree PER; within-episode **ε-ramp**; escalating **stall penalty**; `lava_death=0`     | stall penalty **backfired** (n-step smeared it onto good actions) → removed; ε-ramp + `lava_death=0` kept                        |
| **`160743`** | ε→0.05 with **~200k low-ε tail**, per-bucket water, ε-ramp, PER, 900k                     | **FIRST nonzero greedy: 10%** (key92 door86 right76 water26 lava14 goal10). The **low-ε tail** consolidated greedy               |
| ε-sweep      | fixed-ε eval on `160743`                                                                  | greedy **11%**, **15% @ ε=0.05** (best), 14% @0.10, 9% @0.15. No longer trap-bound (key 91%); ceiling = the funnel               |
| regression   | added `water_pickup_once` (reward only 1st of 3 buckets)                                  | **ferry → 0%**: per-bucket roaming was **load-bearing exploration**. **Reverted** to per-bucket                                  |
| variance     | `n_step=8` + big rewards (goal 300)                                                       | **regressed** — high-magnitude n-step targets inflated value variance, destabilised the tail                                     |
| friend-adopt | peer's **moderate** magnitudes (goal 160, water 30, lava 50, `lava_death` 8), `n_step=5`  | peer reached **28% greedy**; matching magnitudes closed most of our 14%→28% gap                                                  |
| **4M run**   | final config (moderate magnitudes, `n_step=5`, PER, ε-ramp, 700k→0.075 + 3.3M low-ε tail) | **52% greedy** (key96 door94 right90 water62 lava54 goal52); train `succ20` still trending up at 4M → **resumed** for more steps |

### 5.4 REINFORCE — entropy collapse (SimpleRoom)

**Symptom:** 300k steps, eval **4%**; training `succ20` never climbed (~7–9%), episode
length pinned near `max_steps`, **`loss → -0.0`** for the whole tail.

**Confirmed mechanism:** the saved policy outputs `p = [0, 0, 1.0]` on **every** state
(entropy exactly 0) — it degenerated to a **single-action policy**. Once `p(a)=1`,
`log p ≈ 0` and entropy `≈ 0`, so gradients vanish and learning stalls.

**Why:** vanilla Monte-Carlo policy gradient with **no value baseline, no trust region**,
made fragile by (1) weak entropy bonus `0.01`, (2) high `lr=7e-4`, (3) return
normalization on mostly-failure batches manufacturing unit-variance noise → the policy
chased noise into a corner.

**Fix applied** (defaults hardened): `entropy_coef 0.01→0.05`, `lr 7e-4→3e-4`,
`episodes_per_update 8→16`. Re-run pending. **Report angle:** REINFORCE's high variance /
premature convergence is exactly the contrast that motivates a baseline (→ actor-critic).

### 5.5 PPO — greedy-vs-stochastic gap (SimpleRoom)

**Symptom:** 300k steps; training `succ20` 50–85% (plateaued, not 100%), but **greedy eval
and rollout video = all failures (4–6%)**.

**Confirmed mechanism (NOT collapse — entropy healthy at 0.37):** action probs ≈
`[0.001, 0.28, 0.72]`; the argmax is "forward" almost everywhere and the **turn action is
never the argmax**. So the policy navigates by **sampling** occasional turns; greedy =
always-forward → walks into a wall → times out (len 94 vs 52 stochastic).

**Root cause:** **under-convergence** — 300k ÷ (128×8) ≈ **~290 PPO updates**; the policy
found a working _stochastic_ behavior but never sharpened "turn at the wall" into the
argmax.

**Fixes applied:** (1) **linear LR annealing → 0** (CleanRL-style, `anneal_lr=True`) so
the policy commits to deterministic actions late; (2) default SimpleRoom budget **300k →
1M**. **Report angle:** greedy-vs-stochastic gap is a general deployment pitfall — the
training (stochastic) metric overstates the deployable (greedy) policy.

**RESOLVED (fast-convergence PPO).** A friend's proven recipe pointed at the real levers,
now implemented in our PPO + `scripts/train_ppo_simple.py`: **100% greedy eval, converges to
100% training `succ20` by ~23k steps** (first 90% at ~10k), mean solve length ~10 — matching
the fast DQN. What actually fixed it (beyond LR-anneal, which alone was too slow):

1. **Orthogonal init with a tiny (0.01) policy-head gain** — near-uniform initial policy;
   the policy then sharpens to a _deterministic_ optimum instead of relying on sampling.
   This is the single biggest fix for the greedy gap.
2. **Entropy annealing** (0.01 → 0.002 over ~15k) — the PPO analogue of ε-decay: explore
   while discovering, consolidate greedily after (annealing to _0_ collapses prematurely).
3. **Reward scaling** (×0.1; shaped goal ≈50 → value targets ~O(5)) — a stable critic.
4. **Truncation bootstrap** — SAME_STEP autoreset + `γ·V(true final obs)` on timeouts. In
   SimpleRoom nearly every early episode times out at 100 steps; treating those as _true_
   terminals (V=0) systematically poisons the critic. This was the correctness bug that
   kept our first attempt ~2× slower than the friend's.
5. **Small frequent rollouts** (32×8 = 256) + **lr 6e-4** — many updates/step, the driver
   of fast convergence on this short-horizon dense task. γ=0.95, 5 epochs, clip 0.2.

---

## 6. Key findings / lessons (report-ready)

- **Greedy-vs-exploratory gap is the through-line.** DQN (permanent ε-floor → 0% greedy),
  PPO (stochastic 76% vs greedy 6%), REINFORCE (entropy collapse) all show the training
  metric overstating the deployable greedy policy. Fixes differ per family: DQN needs a
  **low-ε consolidation tail**; PPO needs **LR annealing + more updates**; REINFORCE needs
  **entropy pressure + lower lr**.
- **The low-ε consolidation tail fixes DQN's greedy gap.** Annealing ε to ~0.05 then
  training more _at that floor_ produced the first nonzero greedy eval (10%); a permanent
  0.20 floor stayed at **0% greedy**. Post-tail the greedy policy is no longer trap-bound
  (key 91% vs 72%).
- **The remaining DQN ceiling is a multiplicative funnel, not traps.** Greedy cascade
  `right 70% → water 28% → lava 13% → goal 11%`, each stage ~halving; per-stage quality
  compounds into a low end-to-end rate.
- **"Hoard-as-exploration" (subtle, load-bearing):** rewarding **each** of the 3 water
  balls keeps the agent roaming the lava corner with water in hand — which is _how the rare
  toggle-extinguish gets discovered_. A "cleaner" first-bucket-only reward removed that
  roaming and the ferry **regressed to 0%**. A confound that was secretly essential.
- **Reward magnitude × n-step is a variance multiplier.** Big rewards (goal 300) + `n_step=8`
  inflated target variance and regressed the tail; **moderate magnitudes + `n_step=5` were
  strictly better (28%)**. The win was reward _scale discipline_, not a new mechanism.
- **Anti-camp stall penalty backfired** — with n-step it smeared onto necessary preceding
  actions (e.g. "dropping the key is bad") and never stopped the camp. Replaced by the
  **within-episode ε-ramp** (near-greedy through the learned door, exploratory on the
  unlearned frontier). _(Note: the ε-ramp is now **progress-gated** — off while global ε is
  high so it doesn't starve early-chain exploration, phasing in as ε anneals.)_
- **Capacity is not the bottleneck past `width_mult=2`.** A deeper/bigger net does not
  converge the deadly-triad ferry to 100%; PPO (on-policy, no bootstrap-off-replay) is the
  better tool for the tail — the motivation for the actor-critic method.
- **Throughput reality:** tiny nets → low GPU RAM is normal; wall-clock is update-bound.
  `train_freq` and avoiding host syncs matter more than GPU memory.
- **Obs:** single RGB frame sufficient; frame-stacking hurt; grid-aligned `feat_hw=8×8`.

### What we tried that FAILED (valuable negative results)

- Permanent ε floor (0.20) → 0% greedy.
- Frame-stacking → hurt the door skill (57% → 10%).
- `water_pickup_once` (first bucket only) → ferry regressed to 0%.
- Escalating stall penalty → smeared by n-step, backfired.
- `n_step=8` + big rewards → variance regression.
- PER on top of aggressive `reward_scale` (naive sum-tree) → recency bias; needed max-tree.
- REINFORCE with `entropy_coef=0.01` + `lr=7e-4` → entropy collapse to one action.
- PPO 300k / constant LR → greedy policy under-converged (can't turn deterministically).

---

## 7. Best settings (current)

- **DQN / SimpleRoom:** RGB or gray `64×64`; `lr≈2.8e-4`, `γ=0.95`, `batch=128`,
  `train_freq=1`, `τ=0.001`, `eps_end=0.15`, `eps_decay=20k`, `learning_starts=1k`,
  Double DQN → ~90–100% greedy.
- **DQN / ComplexEnv (4M final):** RGB `64×64`, `width_mult=2`, `n_extra_conv=1`,
  `n_step=5`, `lr=2.5e-4`, `γ=0.99`, `reward_scale=0.02`, `train_freq=4`, `τ=0.005`,
  PER (max-tree), `n_envs=8`, ε `1.0→0.075` over 700k + ~3.3M low-ε tail, progress-gated
  within-episode ε-ramp; shaping magnitudes in §3.
- **REINFORCE (hardened):** `entropy_coef=0.05`, `lr=3e-4`, `episodes_per_update=16`,
  `γ=0.99`, return-normalized, entropy bonus.
- **PPO:** `lr=2.5e-4` (**annealed → 0**), `γ=0.99`, GAE `λ=0.95`, `clip=0.2`,
  `ent_coef=0.01`, `vf_coef=0.5`, `rollout=128`, `epochs=4`, `minibatches=4`, `n_envs=8`;
  SimpleRoom budget 1M.

---

## 8. Remaining work (open TODO)

**Training / results**

- [ ] DQN 4M ComplexEnv run — let finish; record final greedy % + stage cascade.
- [ ] REINFORCE — real runs on **both** envs with hardened defaults; confirm no collapse.
- [ ] PPO — real runs on **both** envs (LR-anneal + longer); close the greedy gap on
      SimpleRoom, then ComplexEnv.
- [ ] All required plots for **every algorithm × env** (return, length, success, loss,
      cumulative-steps).
- [ ] ComplexEnv stage-progress analysis (cascade rates) per algorithm.

**Inference / comparison**

- [ ] Greedy (+ small-ε) eval table: avg return, avg steps, success rate — all algos ×
      both envs, fresh seeds; ComplexEnv stage depth.
- [ ] Comparison table and/or grouped bar chart.

**Videos**

- [ ] Mid-training + converged clips for each algo worth showing (grid rollout).

**Discussion / report**

- [ ] Strengths/weaknesses per family × env; shaping that helped vs unintended incentives.
- [ ] Dedicated "best training & inference settings" section.
- [ ] Extension for perfect score (candidate: PER max-tree / ε-ramp ablation write-up).
- [ ] `report_ID1_ID2.pdf` ≤ 12 pages, self-contained; `details.txt` (Colab + IDs);
      optional `explainer.txt`; notebook/Colab with all outputs; seeds + all hypers.

**Codebase maintenance (from the cleanup audit)**

- [x] Run outputs live under `output/` (gitignored). Curated report assets (graph images +
      rollout videos, no weights/history) go into `results/<run>/{graphs,videos}/` — tracked
      — via `scripts/export_results.py <output-run-dir>`.
- [x] DQN supports **resume**: `scripts/train_dqn_complex.py --resume <run-dir> --steps <new-total>`
      reloads weights + counters and continues in place (merged history/plots). Replay buffer
      isn't persisted → ~`learning_starts` warmup on resume.
- [x] Six consistent train scripts: `scripts/train_<algo>_<simple|complex>.py` (DQN/PPO/
      REINFORCE × both envs). PPO/REINFORCE/DQN-simple share `scripts/_algo_cli.py` +
      `scripts/_common.run_training`; `train_dqn_complex.py` keeps its bespoke tuned run.
- [ ] Remove dormant RND path (`archive/rnd.py` move + drop `rnd_coef` in DQN/script).
- [ ] Prune old `output/` sweeps (was `graphs/`, ~815 MB) to leaderboards + winners.
- [ ] Unify DQN trainer: fold `scripts/train_dqn_complex.py` onto
      `scripts/_common.run_training` (do after the DQN runs finish).

---

## 9. Hard constraints (do not violate)

1. Implement algorithms yourself — no SB3 / RLlib / Tianshou / CleanRL.
2. Network input = **pixels only**; no getters/positions as features.
3. No distance- or geometry-based rewards (events only).
4. Env classes fixed except `max_steps`.
5. Seed Python, NumPy, and torch.
