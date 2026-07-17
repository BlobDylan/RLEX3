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
- **Goal:** Fix greedy spin by keeping red agent / green goal channels
- **Setup:** same as Exp4 but **no grayscale** → `(64, 64, 3)`
- **Results:**
  - Train similar to Exp4: `succ20` peaks ~50–70%, late ~35% at low ε
  - No clear improvement over gray (user: “didn’t improve anything”)
- **Notes:** Freeze obs pipeline; next leverage is **model / learning rule**, not more pixel tricks.
