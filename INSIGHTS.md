# MonoGS per-iteration tracking instrumentation + gradient-reuse experiment

Stock MonoGS (`splatonic-monogs`, SPLATONIC sparse-tracking disabled — plain baseline behavior),
instrumented at every tracking iteration of every frame, run on A40 across three scenes:
Replica `room0`, Replica `room1`, TUM `fr1_desk`. Goal: characterize per-iteration tracking
dynamics, then test whether a specific optimization (skip gradient computation on alternate
iterations, reuse the previous iteration's gradient instead) is viable.

## Files in `csv/`

Six CSVs, one row per tracking iteration. Columns:

| Column | Meaning |
|---|---|
| `frame_idx` | Frame index in the sequence |
| `n_gaussians` | Gaussians in the (frozen, during tracking) map |
| `tracking_itr` | Iteration index within this frame (0-based) |
| `grad_computed` | `True` = real render+backward this iteration; `False` = gradient reused (reuse experiment only, always `True` in baseline files) |
| `loss_tracking` | Photometric+geometric tracking loss (scalar) |
| `grad_norm_rot` / `grad_norm_trans` | L2 norm of the rotation/translation pose-delta gradient |
| `grad_cosine_sim` | Cosine similarity between this iteration's full 6D gradient vector `[rot(3), trans(3)]` and the previous iteration's (`null` on each frame's first iteration) |
| `tau_norm` | Magnitude of the Adam step actually applied this iteration (`‖[trans_delta, rot_delta]‖`) |
| `ang_err_deg` / `t_err_cm` | Rotation/translation error vs. ground-truth pose, computed fresh every iteration (not the periodic trajectory-aligned ATE) |
| `converged` | Whether this iteration's step magnitude fell below MonoGS's `1e-4` threshold (frame's tracking loop exits after this) |
| `n_visible` | Gaussians with nonzero screen-space radius this render |
| `iter_wall_ms` | Wall-clock time for this iteration |

Baseline files (`*_trace_baseline.csv`): every iteration is a real gradient computation.
Reuse files (`*_grad_reuse.csv`): even iterations compute a real gradient; odd iterations reuse
the previous iteration's gradient verbatim (no render/loss/backward that step).

## Finding 1 — the "jump" in `tau_norm` is not a real anomaly

Every scene shows a large number of iterations hitting the exact same `tau_norm` value
(~0.00548), almost always at `tracking_itr=0` or `1`. This is not organic signal — it's
Adam's provably deterministic first-step magnitude: for any parameter, Adam's bias-corrected
first update is exactly `±lr` regardless of gradient magnitude. With `lr_rot=0.003` (x3 dims)
and `lr_trans=0.001` (x3 dims), the combined L2 norm is `sqrt(3×0.003² + 3×0.001²) = 0.00548` —
matches the observed spike exactly. It happens once (sometimes twice) per frame, at the start of
every frame's tracking loop, and tells you nothing about how "hard" that frame is.

## Finding 2 — real per-frame difficulty signal: iterations needed to converge

| Scene | Median iters/frame | % of frames hitting the 100-iter cap (never converge) |
|---|---:|---:|
| room0 | 77 | 21% |
| room1 | 75 | 17% |
| fr1_desk | 77 | 12% |

Frames that hit the cap on Replica still converge to reasonable accuracy (~0.5-1cm). On
fr1_desk, capped frames are genuinely bad (5-9cm translation error) — real per-frame hard cases,
likely fast motion / motion blur segments.

## Finding 3 — gradient magnitude is noisy iteration-to-iteration, but that's not the right test

Consecutive-iteration relative change in gradient *norm* (skipping the Adam-warmup zone):
median ~28-34% for both rotation and translation gradients, only ~1-in-5 iterations under 10%
change. Loss itself is much smoother (median 0.3-4%). Magnitude volatility alone can't tell you
whether a stale gradient is safe to reuse, because magnitude and *direction* are independent —
that's what motivated capturing `grad_cosine_sim`.

## Finding 4 — gradient direction stability (fr1_desk, 45,555 iteration-pairs)

| | |
|---|---:|
| Median cosine similarity | 0.839 |
| % with cos_sim > 0.9 (near-identical direction) | 40.7% |
| % with cos_sim < 0 (direction flipped >90°) | 12.2% |
| % with cos_sim < -0.5 (nearly opposite) | 4.1% |

Counter-intuitively, direction stability does **not** improve as tracking approaches
convergence — late iterations (21+) are the *least* stable bucket (36.9% >0.9) versus mid
iterations (6-20: 52.1% >0.9). Plausible explanation: near convergence the true gradient signal
is small, so it's more dominated by noise in a flat region of the loss surface, even though Adam
is still taking meaningful steps.

## Finding 5 — the gradient-reuse experiment: works on TUM, fails badly on Replica

Real experiment (not simulated from the baseline trace — reuse changes the trajectory, so it had
to be actually run): compute a real gradient on even tracking iterations, reuse the previous
iteration's gradient on odd iterations (skipping render+loss+backward on those).

| Scene | Baseline ATE | Reuse ATE | Accuracy | Baseline wall-clock | Reuse wall-clock | Speedup |
|---|---:|---:|---|---:|---:|---:|
| fr1_desk | 5.98-6.62cm | 6.03cm | ~no change | 19m34s | 17m59s | **~8%** |
| room0 | 0.68cm | **3.37cm** | **5x worse** | 1h30m42s | 1h28m42s | **~2% (noise)** |
| room1 | 0.67cm | **4.98cm** | **7.4x worse** | 1h21m13s | 1h12m00s | **~11%** |

Mechanism (fr1_desk): a reused iteration is genuinely 2.92x cheaper than a computed one
(4.61ms vs 13.47ms mean — render+backward dominates cost) — but the scheme also needs ~30% more
total iterations to converge (59,055 vs 45,555 rows), since every other step is a worse
approximation. Net: 1.49x cheaper average iteration cost outpaces the 1.30x more iterations
needed, giving a genuine but modest ~8-12% speedup.

On Replica, the trade is far worse. `room0_grad_reuse.csv` has 199,894 rows out of a theoretical
max of 200,000 (2000 frames × 100 iters) — **nearly every frame hits the iteration cap without
converging**. The reused gradients are bad enough approximations on this scene that the optimizer
needs the full iteration budget and still lands 5x further from ground truth, while barely saving
any wall-clock time at all (the extra iterations needed almost completely eat the per-iteration
savings). `room1` gets a real 11% speedup but still isn't worth a 7.4x accuracy hit.

**Bottom line**: naive fixed-schedule (every-other-iteration) gradient reuse is not a viable
general-purpose optimization for MonoGS tracking. It only works acceptably on the one scene
(fr1_desk, real-world TUM capture) where gradient direction happens to be stable enough across
iterations; it fails on synthetic Replica scenes (room0, room1) where direction changes too much
for a stale gradient to still point the optimizer the right way. Any version of this idea would
need to be scene-adaptive (e.g., only reuse when recent cosine similarity has been consistently
high, always force a fresh gradient after any large direction change) rather than a fixed
skip-every-other-iteration schedule — and even then, room0's near-zero speedup ceiling suggests
the underlying "gradients are locally redundant" assumption may just not hold on that class of
scene.

## Implementation notes

- Instrumentation lives in `splatonic-monogs/utils/tracking_trace.py` (new file) and is wired
  into `splatonic-monogs/utils/slam_frontend.py`'s `FrontEnd.tracking()` method, gated behind
  `SPLATONIC_TRACKING_TRACE=1` — zero cost when unset.
- The gradient-reuse mode is gated behind `SPLATONIC_GRAD_REUSE_ALTERNATE=1` in the same file.
- All runs used stock/baseline MonoGS behavior (`use_splatonic=False` — the SPLATONIC
  sparse-tracking patch that repo also supports was not involved in any of this).
- Hardware: A40 (`--partition=a40 --qos=a40_4d --account=a40`), a separate venv from this
  session's usual A100 jobs (`TORCH_CUDA_ARCH_LIST=8.6` vs `8.0`).
