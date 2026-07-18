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
  - `max_steps=200` (override env default 400 — more episodes/budget, less step-penalty burn)
  - Wrapper: `ComplexShapingWrapper` (first-time / per-spawn-tile anti-farm)
- **Locked rewards:**
  - `goal_scale=50`, `step_penalty=0.1` (match SimpleRoom)
  - `key_pickup=+5`, `door_open=+10`, `enter_right_room=+5` — once/episode
  - `water_pickup=+2` — once per original spawn tile
  - `lava_extinguish=+5`/tile, `lava_death=-10`
- **Results:** random 30 eps sanity OK — `key_pickup` 27/30, `door_open` 2/30, `enter_right` 1/30
- **Notes:** **6a DONE.** Next = 6b stage metrics + short DQN smoke.

## Experiment 9 (Step 6b): ComplexEnv DQN smoke @ 10k

- **Date:** 2026-07-17
- **Goal:** Wire stage metrics into DQN logs; short train smoke (not a full solve)
- **Setup:** `make_complex_env()` (gray, actions 0–5, `max_steps=200`, locked 6a shaping)
- **Hyperparameters:** SimpleRoom Exp6 winner (`lr=2.81e-4`, `γ=0.95`, `batch=128`, …)
- **Results (local MPS, longer than planned smoke ~50k+ / ~54 min):**
  - Stage logs work: `key≈45–75%`, `door≈0–10%` sporadic, right/water/lava/goal ≈ 0
  - Returns stuck ~−15 to −20 (mostly step penalty / no mid-stage bonuses)
  - **Bottleneck:** open door after key (expected for early ComplexEnv)
- **Notes:** 6b wiring OK. Prefer **Colab GPU** for 6c long trains. Door unlock is next learning target.

## Experiment 10: Throughput — GPU “empty” RAM is normal; train was update-bound

- **Date:** 2026-07-17
- **Goal:** Explain Colab `GPU RAM 0.3/15 GB` and speed up DQN wall-clock
- **Finding:** Q-net is ~0.5M params (~2 MB). Low GPU memory ≠ “not using GPU”. Networks already on `cuda`/`mps`. Wall-clock was dominated by **1 SGD step per env step** (`train_freq=1`) + `loss.item()` sync every update (MPS: ~45–65 ms/update vs ~0.4 ms `env.step`).
- **Changes in `algorithms/dqn.py`:**
  - `log_loss_every` (default 50) — skip host sync most updates
  - `gradient_steps` + `train(..., n_envs=)` via `SyncVectorEnv` + batched `select_actions`
  - CUDA replay: `pin_memory` + `non_blocking`
- **Microbench (local MPS, EmptyEnv proxy, 400 steps):**
  - `train_freq=1`: ~32–36 steps/s
  - `train_freq=4`: ~148 steps/s (**~4–5×**)
  - `n_envs=8` + `train_freq=4`: ~151 steps/s (similar here; helps more when ε is low / on CUDA)
- **Notebook 6b defaults:** `train_freq=4`, `n_envs=8`, smoke `10_000` steps
- **Notes:** For 6c long Colab runs, keep `train_freq=4` (or 8); bump `n_envs` if GPU util still low. Don’t expect GPU RAM to fill — model is tiny.

## Experiment 10b: Still slow? Not CPU — `n_envs=8` over-updated on MPS

- **Date:** 2026-07-17
- **Symptom:** 10k ComplexEnv smoke ≈ **2.5 min** (~67 steps/s) with `n_envs=8`, `train_freq=4`
- **CPU check:** full ComplexEnv stack ≈ **0.4 ms/step** (~2400 steps/s). CPU is *not* the bottleneck; Activity Monitor may look busy from Python + MPS driver, but env render is cheap.
- **Cause:** SyncVectorEnv runs 8 envs **serially**, then the old vec learn loop fired **2 SGD updates per vector step** (8/4). MPS update ≈ 14–45 ms → wall-clock still update-bound. CPU DQN updates are *worse* (~470 ms @ bs128) — stay on MPS.
- **Fixes:**
  - Vec learn: **one** update burst per `train_freq` threshold (not `n_envs/train_freq` bursts)
  - Smoke defaults → `n_envs=1`, `train_freq=8`, `batch_size=64`, hard target `tau=1` / `target_update_freq=500`
- **Bench:** `mps freq=8 n=1 bs=64` ≈ **400 steps/s**. Confirmed: full 10k smoke ≈ **35s** on MPS (~286 steps/s end-to-end with logging/eval).
- **Colab tip:** CUDA updates are faster → `n_envs=4..8` can help again; Mac MPS: prefer `n_envs=1` + higher `train_freq`.
- **Notes:** **6b DONE.** Next = 6c longer train + graphs/videos.

## Experiment 12: Overnight ComplexEnv sweep (door-focused)

- **Date:** 2026-07-17 → 2026-07-18
- **Motivation:** Exp11 — longer same-hparams run stuck at door ≈0%
- **Runner:** `scripts/overnight_complex.py` → `graphs/overnight_complex/`
- **Status @ morning:** **41/47** finished (still on Phase F); **no goal successes**
- **Headline:** shaping/hparams alone usually fail like Exp11 — **except `train_freq=2` (frequent_updates)**, which finally moves the door stage.
- **Winners (by late `stage_door`):**
  1. **`E_deep_frequent`** — door_late **46%**, right_late **42%**, eval_door **47%** (250k, door_heavy, `train_freq=2`)
  2. **`B_dqn_frequent_updates`** — door_late **44%** already at 80k (same preset)
  3. **`E_deep_rgb_extreme`** — door_late **32%**, right **22%** (RGB + door_extreme + very slow ε)
- **Most other trials:** key 30–80%, door ≤6% — confirms Exp11
- **Caveat:** greedy diagnose on winners often **collapses** (drop-spam or forward-only) even when train door↑ — need longer train / ε care / anti-collapse before trusting eval
- **Next (6d):** promote `frequent_updates` + door_heavy (try RGB too); push water/lava; fix greedy collapse
- **Monitor:** `graphs/overnight_complex/LEADERBOARD.md`

## Experiment 13: 2×2 tile_size × network width (on winner recipe)

- **Date:** 2026-07-18
- **Recipe locked:** `frequent_updates` (`train_freq=2`) + `door_heavy` + 250k steps, seed=42
- **Grid:**
  | | old arch `width_mult=1` (~0.47M) | new arch `width_mult=2` (~1.9M) |
  |---|---|---|
  | **tile=12** | `tile12_arch1` (overnight baseline class) | `tile12_arch2` |
  | **tile=16** | `tile16_arch1` | `tile16_arch2` |
- **Runner:** `scripts/ab_tile16.py` → `graphs/ab_tile_arch/`
- **Note:** CNN input stays `64×64`; tile=16 only changes render resolution *before* crop/inset/resize (sharper nearest downsample).
- **Partial results:** `tile12_arch2` best late door (56%) vs arch1 (38%); same eval_door 27%. `tile16_arch1` weak (eval_door 7%). `tile16_arch2` mid-run looked strong (door~70% @100k) but unfinished.
- **Notebook lock (user):** `tile_size=12`, **no `TileInsetWrapper`**, `frequent_updates` + `door_heavy`. Overnight removed from notebook.
- **Arch sweep (6c):** DQN now supports `width_mult` / `n_extra_conv` / `fc_mult`. Notebook `_ARCH_SWEEP` = w2 (~1.9M), w3 (~4.2M), w4 (~7.4M), w3_deep, w3_widehead → `graphs/complex_arch_sweep_dqn/` via `RUN["complex_train"]`.
- **Rainbow DQN:** `algorithms/rainbow.py` — Double + Dueling + NoisyNets + n-step(3) + PER + C51(51). Notebook: `_USE_RAINBOW=True` → `graphs/complex_arch_sweep_rainbow/` with `rb_w1` only.
- **Leave-right / key-drop (6d):** `leave_right_room` −every backtrack; `enter_right_room` +first entry; `key_drop` +first drop **in right room**; `key_drop_locked_left` −every drop in left while door locked. `door_heavy`: enter/leave ±8, key_drop +10, key_drop_locked_left −8.

## Experiment 11: Colab CUDA long run (~132k) — stuck at door

## Experiment 11: Colab CUDA long run (~132k) — stuck at door

- **Date:** 2026-07-17
- **Goal:** See if more steps alone unlock ComplexEnv mid-stages
- **Setup:** gray `64×64×1`, actions 0–5, CUDA, SimpleRoom-ish hypers (`eps→0.15` by ~20k)
- **Results (@132k / 660 eps):**
  - `succ20=0%` entire run; goal/water/lava/right = **0%**
  - `key` oscillates **20–70%** (no upward trend; sometimes *worse* late)
  - `door` almost always **0%**, rare **5%** blips — never learns to open
  - `mean20` return stuck **≈ −16…−19** (step-penalty dominated)
- **Interpretation:** **Key is reachable by chance; door is the hard bottleneck.** More of the same budget/hparams will not finish the task. Overnight 6c must vary **shaping (door-heavy)**, **ε schedule (slower decay)**, and DQN knobs — score trials by `stage_door` / `stage_right`, not goal alone.
- **Notes:** Feeds Step 6c/6d design. Do not treat Exp11 as a failed algorithm — treat it as a **stage diagnosis**.

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

