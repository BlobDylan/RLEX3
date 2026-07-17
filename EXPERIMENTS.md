# Experiments

Log of experiments tried for the RL final project — setups, hyperparameters, results, and notes.

Plots and figures live in [`graphs/`](graphs/).

---

## Experiment 1: DQN SimpleRoom — sparse smoke (baseline)

- **Date:** 2026-07-17
- **Goal:** Sanity-check DQN pixel pipeline on `SimpleRoomEnv`
- **Setup / algorithm:** DQN, grayscale `40×40×1`, actions `(0,1,2)`, sparse `+1`/`0`
- **Hyperparameters:** `5_000` steps, `lr=1e-4`, `γ=0.99`, ε `1.0→0.05` over `20_000`
- **Results:** mean20 ≈ 0.1–0.15; greedy eval return `0.0`, length `100` (no goal)
- **Graphs:** —
- **Notes / next steps:** Too short + sparse. Added `SimpleRoomShapingWrapper` + success logging for next run.

## Experiment 2: DQN SimpleRoom — shaped + 50k steps

- **Date:** 2026-07-17
- **Goal:** Pass SimpleRoom sanity check with dense shaping
- **Setup / algorithm:** DQN + `SimpleRoomShapingWrapper(goal_scale=50, step_penalty=0.1)` + grayscale + actions `(0,1,2)`
- **Hyperparameters:** `50_000` steps, `lr=1e-4`, `γ=0.99`, ε decay over `50_000`, target update every `500`
- **Results:**
  - Training `succ20` oscillates ~5–40% (occasional wins `return≈43–50`)
  - Greedy eval: **success_rate = 0.0**, mean_return ≈ −10, length 100
- **Notes / next steps:** Exploration finds goal; greedy policy collapses. Led to Exp3 fixes.

## Experiment 3: Double DQN SimpleRoom — soft target + MiniGrid CNN

- **Date:** 2026-07-17
- **Goal:** Fix greedy collapse after Exp2
- **Setup / algorithm:** Double DQN, soft target `τ=0.005`, bootstrap on `terminated` only, smaller CNN, shaping unchanged
- **Hyperparameters:** `50_000` steps, `lr=2.5e-4`, `train_freq=1`, ε `1.0→0.05` over `50k`
- **Results:**
  - Late train: `succ20` up to ~60–70%
  - Greedy diagnose: **almost always `forward`** (`action_counts={0:1, 1:0, 2:99}`)
  - Eval: **success_rate = 0.4**
- **Notes:** Forward-collapse. Led to obs readability work.

## Experiment 4: Double DQN + wall-crop + tile-inset + resize64 (gray)

- **Date:** 2026-07-17
- **Goal:** Better readable obs (tile=12, inset 75%, gray→64)
- **Setup / algorithm:** Double DQN, soft τ, shaping `+50/−0.1`, actions `(0,1,2)`
- **Hyperparameters:** `50_000` steps, `lr=2.5e-4`, `train_freq=4`, ε `1.0→0.05`
- **Results:**
  - Train `succ20` peaks ~70–80%
  - Greedy: **spin** `{0:47, 1:47, 2:6}` → eval **success_rate = 0.2**
- **Notes:** Training OK; greedy spin. Try RGB next.

## Experiment 5: Exp4 pipeline but RGB

- **Date:** 2026-07-17
- **Goal:** Keep red agent / green goal channels
- **Setup:** same as Exp4 but `(64, 64, 3)` — no grayscale
- **Results:** train similar to Exp4; no clear greedy improvement
- **Notes:** Freeze obs; try hyperparam search next.

## Experiment 6: Random hyperparam search (8 × 20k)

- **Date:** 2026-07-17
- **Goal:** Find hypers that produce a non-spinning greedy policy
- **Setup:** frozen RGB pipeline; Double DQN; seed=0; `n_trials=8`, `steps_per_trial=20_000`
- **Results:**
  - Trial 4 winner: greedy **success_rate=0.90**, mean_length≈19.3
  - Winner hypers: `lr≈2.81e-4`, `γ=0.95`, `batch=128`, `train_freq=1`,
    `τ=0.001`, `eps_end=0.15`, `eps_decay=20k`, `learning_starts=1k`
- **Notes:** Led to Exp6b full retrain.

## Experiment 6b: Full 50k retrain with Exp6 winner

- **Date:** 2026-07-17
- **Goal:** Confirm SimpleRoom sanity check with best hypers
- **Setup:** RGB pipeline frozen; winner hypers from Exp6 trial 4
- **Hyperparameters:** `lr=2.81e-4`, `γ=0.95`, `batch=128`, `train_freq=1`, `τ=0.001`,
  `eps_end=0.15`, `eps_decay=20k`, `learning_starts=1k`, Double DQN
- **Results:**
  - Wall time ~**47 min** (MPS)
  - `succ20` reaches **100% by ~ep 220** and **stays at 100%** through the rest of training
  - `mean20` ≈ **48.5–49** (near max shaped return); episodes short / decisive
- **Notes:** **Step 3 SimpleRoom DQN sanity check: PASSED** (training clearly solved). Keep these as the SimpleRoom baseline going forward.

## Experiment 8 (Step 6a): ComplexEnv stack — shaping locked

- **Date:** 2026-07-17
- **Goal:** Wire ComplexEnv factory before any long DQN train
- **Setup:**
  - Obs: `tile=12 → wall-crop → inset 0.75 → grayscale → 64×64×1`
  - Actions: `(0..5)` — **no `done`**
  - Wrapper: `ComplexShapingWrapper` (first-time / per-spawn-tile anti-farm)
- **Locked rewards:**
  - `goal_scale=50`, `step_penalty=0.1` (match SimpleRoom)
  - `key_pickup=+5`, `door_open=+10`, `enter_right_room=+5` — once/episode
  - `water_pickup=+2` — once per original spawn tile
  - `lava_extinguish=+5`/tile, `lava_death=-10`
- **Results:** random 30 eps sanity OK — `key_pickup` 27/30, `door_open` 2/30, `enter_right` 1/30
- **Notes:** **6a DONE.** Next = 6b stage metrics + short DQN smoke.

## Experiment 7: Grayscale A/B with Exp6 winner hypers

- **Date:** 2026-07-17
- **Goal:** Test whether Exp4 gray-spin was hypers vs color; keep RGB winner hypers, switch to grayscale
- **Setup:** same stack as Exp6b but `GrayscaleWrapper` before resize → `(64, 64, 1)`; **20k** steps
- **Hyperparameters:** same as Exp6 winner / Exp6b
- **Results:**
  - Train: `succ20` → **100%** by ~ep 280 (stable ~95–100% after ~ep 220)
  - Greedy diagnose: **success**, actions `{0:1, 1:1, 2:9}` (forward-dominant, **no spin**)
  - Eval (20 eps): **success_rate=0.90**, mean_length≈19.3, mean_return≈43.1
  - Video: `videos/dqn_simple_room_gray_exp7.mp4` (11 steps, return 48.9)
- **Notes:** **Grayscale ≈ RGB** under winner hypers. Exp4 spin was hypers/training, not grayscale itself.
  Adopt **grayscale** as SimpleRoom default (cheaper `64×64×1`). ComplexEnv 6a also starts grayscale (user choice).

