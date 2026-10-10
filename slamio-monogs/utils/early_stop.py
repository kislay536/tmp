import json
import os


class EarlyStop:
    """Unified early-stopping for tracking optimization loops.

    Usage per frame:
        stopper = EarlyStop(config.get("early_stop", {}))
        stopper.reset()
        for i in range(max_iters):
            ...compute loss, pose_delta_norm...
            if stopper.check(i, loss_val, pose_delta_norm):
                break
        print(stopper.summary())

    Dynamic retuning (optional):
        Add to the early_stop config:
            retune_every:  100   # re-tune every N frames (0 = disabled)
            retune_frames: 10    # probe frames per retune window

        Every `retune_every` frames, early stopping is disabled for
        `retune_frames` frames to collect fresh signals.  The analysis
        functions from tune_early_stop.py are then called directly (pure
        numpy, no subprocess) to update loss_eps, pose_eps, min_iters.

    Signal logging (for offline threshold tuning):
        Add "log_signals": "/abs/path/signals.jsonl" to the early_stop config.
    """

    def __init__(self, cfg: dict):
        self.enabled   = bool( cfg.get("enabled",   False))
        self.min_iters = int(  cfg.get("min_iters",  10))
        self.loss_eps  = float(cfg.get("loss_eps",   1e-4))
        self.patience  = int(  cfg.get("patience",   3))
        self.pose_eps  = float(cfg.get("pose_eps",   1e-4))

        # ── which statistic decides "converged" ──────────────────────────────
        #
        # "consecutive" (default) is the original rule: `patience` consecutive
        # single-step relative changes below loss_eps. It is a RUN-LENGTH
        # COUNTER ON A ONE-STEP DIFFERENCE, which is the statistic most
        # exposed to per-iteration noise - one jittery step resets it, one
        # quiet step completes it.
        #
        # That matters here because the CUDA iteration graph appears to have
        # CHANGED the noise. The measured signature is exactly what less
        # jitter predicts: the fire rate went 64% -> 71.8% while the AVERAGE
        # firing iteration stayed put (93 -> 94). No frame stops sooner; more
        # frames manage to string `patience` quiet steps together. The
        # rasterizer's backward accumulates with atomicAdd, so the loss
        # carries ordering-dependent jitter, and a replay issues an identical
        # pre-baked launch configuration every iteration where eager
        # scheduling varies.
        #
        # If that is right, the consecutive rule is partly measuring HOW the
        # kernels were launched rather than whether the pose converged, and
        # the fix is a statistic that averages the jitter out instead of being
        # triggered by it.
        #
        # "window" is that statistic: total improvement across the last
        # `window` iterations, against a budget of loss_eps per iteration.
        #
        #     (loss[i-W] - loss[i]) / |frame_start_loss|  <  loss_eps * W * slack
        #
        # Real descent is coherent and sums over the window; zero-mean jitter
        # cancels, roughly as sqrt(W). So the decision stops depending on how
        # any single step happened to land, which is what makes it comparable
        # between an eager and a replayed run. It is also signed, not
        # absolute: a loss that has stopped improving is done regardless of
        # which way the last step went, and the committed pose is the
        # best-loss candidate either way.
        #
        # Same units and same fitted loss_eps, so the retuner still applies -
        # tune_early_stop.simulate_stopper takes the same two knobs and must
        # be given them, or the fit optimises a criterion the loop is not
        # running (see _do_retune_inner).
        self.criterion    = str(  cfg.get("criterion", "consecutive"))
        self.window       = int(  cfg.get("window", self.patience))
        self.window_slack = float(cfg.get("window_slack", 1.0))
        if self.criterion not in ("consecutive", "window"):
            raise ValueError(f"early_stop.criterion must be 'consecutive' or "
                             f"'window', got {self.criterion!r}")
        self._loss_hist = []
        self._pose_hist = []
        # Before the first retune, min_iters/loss_eps/pose_eps are whatever
        # the static config says - no real convergence data has been seen
        # yet, and the map is thinnest/least-constrained early in the
        # sequence, so a min_iters tuned for a mature map can cut frames off
        # before the pose has genuinely converged (false plateau, real
        # drift). warmup_min_iters lets that early window be conservative
        # independent of the steady-state min_iters; defaults to min_iters
        # itself (no-op) unless set.
        self.warmup_min_iters = int(cfg.get("warmup_min_iters", self.min_iters))

        self._plateau           = 0
        self._pose_plateau      = 0
        self._prev              = None
        self._frame_start_loss  = None  # loss at iter 0 of each frame — normalization anchor
        self.fires              = 0
        self.total_frames  = 0
        self._fire_iter_sum = 0

        # signal logging
        self._log_path  = cfg.get("log_signals", None)
        self._frame_buf = []
        if self._log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
            open(self._log_path, "w").close()

        # dynamic retuning
        self.retune_every                = int(  cfg.get("retune_every",                0))
        self.retune_frames               = int(  cfg.get("retune_frames",              10))
        self.retune_probe_max_iters_frac = float(cfg.get("retune_probe_max_iters_frac", 0.75))
        self.retune_quality_threshold    = float(cfg.get("retune_quality_threshold",    0.20))
        self.retune_target_iter_frac     = float(cfg.get("retune_target_iter_frac",     0.60))
        self.retune_min_iters_floor      = int(  cfg.get("retune_min_iters_floor",       8))
        # ── stabilising the fit ──────────────────────────────────────────────
        #
        # Measured, profiling/retune_report.py over 10 runs of 3 configs: the
        # fitted loss_eps spans 4.24e-04 to 4.41e-03, a 10x range, and it is
        # not drift - it bounces 2-4x between CONSECUTIVE retunes of the SAME
        # run (7.91e-04 -> 3.22e-03 -> 1.73e-03 -> 1.48e-03), in both
        # directions, with no consistent trend as the map matures. It is an
        # unstable estimator fitted from 15 probe frames, not a signal.
        #
        # And it is the lever that matters. Across two independent arms, all
        # six correlations came out in the same direction:
        #
        #     corr(loss_eps, fire%)  +0.96  +1.00
        #     corr(loss_eps, it/fr)  -0.94  -0.98
        #     corr(loss_eps, ATE)    +0.97  +0.90
        #
        # Looser threshold -> fires more -> fewer iterations -> worse ATE. That
        # is the whole chain behind this config's ATE spread, which the file
        # had been calling "bimodality".
        #
        # Three fixes, each aimed at something the report actually showed:
        #
        # retune_reject_fallback  suggest_loss_eps returns a hard-coded 1e-4
        #   when NO candidate satisfies its rate/speed/quality criteria. 1e-4
        #   is not on its search grid (logspace(-5,-1,60)), so a fitted value
        #   of exactly 1e-4 is unambiguously "the fit gave up" rather than "the
        #   fit chose this". One observed run applied it and ran on a threshold
        #   ~15x tighter than its own previous retune. Keeping the previous
        #   value is strictly better than adopting a sentinel.
        #
        # retune_max_change  cap how far one retune may move loss_eps, as a
        #   ratio. Bounds the excursion without assuming the estimator is
        #   biased. Not applied to the FIRST retune: that one moves from the
        #   static startup value (4e-3, deliberately loose so it can fire
        #   before any data exists) to the first real fit, and that move is
        #   meant to be large.
        #
        # retune_smooth  EMA weight on the new fit, so each retune is a
        #   correction rather than a replacement. 1.0 is the current behaviour
        #   (take the new value outright); 0.5 averages with the previous.
        #
        # All three default OFF so existing numbers are unaffected.
        self.retune_reject_fallback = bool( cfg.get("retune_reject_fallback", False))
        self.retune_max_change      = float(cfg.get("retune_max_change",       0.0))
        self.retune_smooth          = float(cfg.get("retune_smooth",           1.0))
        # ── the speed/accuracy dial ──────────────────────────────────────────
        #
        # A blunt multiplier on every fitted loss_eps. >1 loosens the threshold,
        # so early stopping fires on more frames, each frame runs fewer
        # iterations, and the run gets faster and less accurate; <1 does the
        # reverse. It exists because loss_eps is the ONLY thing measured to move
        # this system along that trade-off in a controlled way:
        #
        #   corr(loss_eps, fire%)  +0.91 .. +1.00   (three arms, two scenes)
        #   corr(loss_eps, it/fr)  -0.75 .. -0.99
        #
        # Use this rather than the knobs that look like they should work:
        #   retune_target_iter_frac  measured NULL - it only feeds the fit,
        #                            which converges wherever the probe data
        #                            leads regardless of the target given to it.
        #   patience                 measured CATASTROPHIC at 5 (ATE 56.63). It
        #                            is a run-length counter on a one-step
        #                            difference, so raising it changes what is
        #                            being measured, not just how much.
        #   min_iters                works (a hard comparison in check()) but 50
        #                            was already tried against 70 and was worse.
        #
        # Applied AFTER the stabilisers and to the first retune as well: it is a
        # deliberate bias, not a smoother. Does not touch the static startup
        # loss_eps, which only governs frames before the first retune.
        #
        # MIND THE STABILITY CLIFF. Below roughly 100 iterations/frame this
        # system sits at the edge of stable convergence and the rasterizer's
        # atomicAdd nondeterminism decides which side a run lands on: 117
        # iters/frame gave ATE 3.38/3.52, 88 iters/frame gave 3.73/3.81/6.49/
        # 8.61 on consistent thresholds. Scales far above ~1.5 are expected to
        # fall off it, and a single run near the edge carries no information.
        self.retune_loss_eps_scale  = float(cfg.get("retune_loss_eps_scale",   1.0))
        # The unscaled trajectory the stabilisers anchor on - see _do_retune_inner.
        self._loss_eps_base = self.loss_eps
        self.dataset_frames              = int(  cfg.get("dataset_frames",               0))
        self._probe_mode                 = False
        self._probe_buf                  = []
        retune_start = int(cfg.get("retune_start", self.retune_every))
        self._frames_since_retune        = self.retune_every - retune_start
        self._retune_count               = 0
        self._max_iters_observed         = 0

        self._logged_first_check = False

        _crit = (f"criterion=window(W={self.window}, slack={self.window_slack})"
                 if self.criterion == "window"
                 else f"criterion=consecutive(patience={self.patience})")
        print(f"[EarlyStop] constructed (enabled={self.enabled}, "
              f"min_iters={self.min_iters}, warmup_min_iters={self.warmup_min_iters}, "
              f"{_crit}, loss_eps={self.loss_eps}, "
              f"pose_eps={self.pose_eps}, retune_every={self.retune_every})", flush=True)

    # ── frame lifecycle ───────────────────────────────────────────────────────

    def reset(self):
        """Call once at the start of each frame's tracking."""
        # flush previous frame
        if self._frame_buf:
            self._max_iters_observed = max(self._max_iters_observed, len(self._frame_buf))
            if self._log_path:
                self._write_frame(self.total_frames - 1, self._frame_buf)
            if self._probe_mode:
                self._probe_buf.append({
                    "frame":   self.total_frames - 1,
                    "signals": list(self._frame_buf),
                })

        self._frame_buf         = []
        self._plateau           = 0
        self._pose_plateau      = 0
        self._prev              = None
        self._frame_start_loss  = None
        self._loss_hist         = []
        self._pose_hist         = []
        self.total_frames      += 1

        # probe window just filled → retune
        if self._probe_mode and len(self._probe_buf) >= self.retune_frames:
            self._do_retune()
            self._probe_mode = False

        # start next probe window if due, but not if too close to end of dataset.
        #
        # `self.enabled` GATES THIS, and it did not used to. check() tests
        # `_probe_mode` before it tests `enabled`, so with early stopping
        # DISABLED the probe still opened and truncated 15 frames to
        # retune_probe_max_iters_frac of their budget - visibly, as
        # "Tracking Time Step: 100: 75%| 30/40" in a run configured for 40.
        #
        # That is wrong twice over. Retuning exists to fit thresholds that only
        # matter when early stopping fires, so with it off the probe collects
        # data nothing will ever use. And it silently perturbs a fixed 15-frame
        # window of every run - which in the dose/preconditioner sweeps, where
        # every config runs a HARD iteration cap on purpose, means the cap was
        # not actually held for 15 of 250 frames.
        if self.enabled and self.retune_every > 0 and not self._probe_mode:
            self._frames_since_retune += 1
            if self._frames_since_retune >= self.retune_every:
                frames_left = (self.dataset_frames - self.total_frames
                               if self.dataset_frames > 0 else float("inf"))
                if frames_left >= self.retune_frames + 10:
                    self._probe_mode = True
                    self._probe_buf  = []
                    self._frames_since_retune = 0
                    print(f"[EarlyStop] Probe window starting at frame {self.total_frames} "
                          f"({self.retune_frames} frames)")
                else:
                    # too close to end — skip this probe window
                    self._frames_since_retune = 0

    def _write_frame(self, frame_idx: int, buf: list):
        with open(self._log_path, "a") as f:
            f.write(json.dumps({"frame": frame_idx, "signals": buf}) + "\n")

    def flush(self):
        """Write the last frame's signals. Call once after all frames are done."""
        if self._frame_buf:
            if self._log_path:
                self._write_frame(self.total_frames - 1, self._frame_buf)
            if self._probe_mode:
                self._probe_buf.append({
                    "frame":   self.total_frames - 1,
                    "signals": list(self._frame_buf),
                })
        self._frame_buf = []

    # ── the convergence statistics ────────────────────────────────────────────

    def _rel(self, a, b):
        """Change from a→b normalised by the frame's starting loss."""
        return abs(a - b) / (abs(self._frame_start_loss) + 1e-10)

    def _record(self, loss_val, pose_delta_norm):
        """Update every plateau statistic for this iteration.

        Called on EVERY iteration including those below min_iters and those in
        a probe window: the counters and the history have to be warm the moment
        the floor lifts, or the first `patience`/`window` iterations after it
        would be judged on an empty record.
        """
        if self._prev is not None:
            self._plateau = (self._plateau + 1
                             if self._rel(self._prev, loss_val) < self.loss_eps else 0)
        self._prev = loss_val
        if pose_delta_norm is not None and self.pose_eps > 0:
            self._pose_plateau = (
                self._pose_plateau + 1 if pose_delta_norm < self.pose_eps else 0
            )
        # The window rule needs loss[i-W], so keep W+1 samples.
        self._loss_hist.append(loss_val)
        if len(self._loss_hist) > self.window + 1:
            self._loss_hist.pop(0)
        if pose_delta_norm is not None:
            self._pose_hist.append(pose_delta_norm)
            if len(self._pose_hist) > self.window:
                self._pose_hist.pop(0)

    def _plateaued(self, pose_active: bool) -> bool:
        """Has the frame converged, by whichever criterion is configured?"""
        if self.criterion == "window":
            # Not enough history yet - min_iters is far above `window` in every
            # tuned config, so this only guards the first iterations of a frame.
            if len(self._loss_hist) <= self.window:
                return False
            improvement = ((self._loss_hist[0] - self._loss_hist[-1])
                           / (abs(self._frame_start_loss) + 1e-10))
            budget = self.loss_eps * self.window * self.window_slack
            loss_done = improvement < budget
            if not pose_active:
                return loss_done
            pose_done = (len(self._pose_hist) >= self.window
                         and sum(self._pose_hist) / len(self._pose_hist) < self.pose_eps)
            return loss_done and pose_done

        if pose_active:
            return self._plateau >= self.patience and self._pose_plateau >= self.patience
        return self._plateau >= self.patience

    # ── per-iteration check ───────────────────────────────────────────────────

    def check(self, iteration: int, loss_val: float, pose_delta_norm: float = None) -> bool:
        """Return True if the loop should stop early."""
        if not self._logged_first_check:
            # Same rationale as AdaptiveMapper's first-call print: silence
            # here is ambiguous between "never called" and "called but
            # never fires before min_iters" - this confirms the calling
            # code's loop actually reached check() at all.
            print(f"[EarlyStop] check() reached for the first time "
                  f"(enabled={self.enabled}, min_iters={self.min_iters})", flush=True)
            self._logged_first_check = True
        if self._log_path or self._probe_mode:
            self._frame_buf.append([iteration, loss_val, pose_delta_norm])

        # record the starting loss for this frame on the first iteration
        if self._frame_start_loss is None:
            self._frame_start_loss = loss_val

        # probe mode: collect signals, stop early once cap is reached
        if self._probe_mode:
            self._record(loss_val, pose_delta_norm)
            # cap probe frame at fraction of observed max_iters (default 75%)
            if (self.retune_probe_max_iters_frac > 0
                    and self._max_iters_observed > 0):
                cap = max(1, round(self.retune_probe_max_iters_frac * self._max_iters_observed))
                if iteration >= cap - 1:
                    return True  # end probe frame early (does not count as a fire)
            return False

        # Use the conservative warmup floor until real convergence data has
        # driven at least one retune; after that, self.min_iters is the
        # data-driven value (or still the static config value, if
        # retuning is disabled).
        _effective_min_iters = (
            self.warmup_min_iters if self._retune_count == 0 else self.min_iters
        )
        if not self.enabled or iteration < _effective_min_iters:
            self._record(loss_val, pose_delta_norm)
            return False

        self._record(loss_val, pose_delta_norm)

        # AND: both must plateau when pose criterion is active; loss-only otherwise
        stop = self._plateaued(pose_delta_norm is not None and self.pose_eps > 0)

        if stop:
            self.fires += 1
            self._fire_iter_sum += iteration + 1
        return stop

    # ── dynamic retuning ──────────────────────────────────────────────────────

    def _do_retune(self):
        """Analyse probe buffer and update thresholds in place (pure numpy, no subprocess)."""
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tune_early_stop import (
                suggest_loss_eps, suggest_pose_eps, compute_pose_tail_mass,
                compute_tail_level, find_elbow,
                _POSE_TAIL_MASS_THRESHOLD,
            )
            import numpy as np
        except Exception as e:
            print(f"[EarlyStop] Retune skipped — import failed: {e}")
            return

        try:
            self._do_retune_inner(
                suggest_loss_eps, suggest_pose_eps, compute_pose_tail_mass,
                compute_tail_level, find_elbow,
                _POSE_TAIL_MASS_THRESHOLD, np,
            )
        except Exception as e:
            import traceback
            print(f"[EarlyStop] Retune failed: {e}")
            traceback.print_exc()

    def _do_retune_inner(self, suggest_loss_eps, suggest_pose_eps, compute_pose_tail_mass,
                         compute_tail_level, find_elbow,
                         _POSE_TAIL_MASS_THRESHOLD, np):

        frames = self._probe_buf
        if not frames:
            return

        max_iters = max(len(fd["signals"]) for fd in frames)
        profile = {
            "quality_threshold": self.retune_quality_threshold,
            "target_iter_frac":  self.retune_target_iter_frac,
            "min_iters_floor":   self.retune_min_iters_floor,
        }

        # suggest min_iters from elbow detection on probe frames
        all_losses = [
            [s[1] for s in fd["signals"]]
            for fd in frames if len(fd["signals"]) > 2
        ]
        if all_losses:
            rel_changes_per = [
                [abs(l[i] - l[i-1]) / (abs(l[i-1]) + 1e-10) for i in range(1, len(l))]
                for l in all_losses
            ]
            tail_levels = [compute_tail_level(rc) for rc in rel_changes_per]
            elbows = [find_elbow(l, tl) for l, tl in zip(all_losses, tail_levels)]
            _raw_elbow_median = int(np.median(elbows))
            new_min_iters = max(profile["min_iters_floor"], _raw_elbow_median)
            # Diagnostic: every retune so far has landed exactly on
            # min_iters_floor (70). Need to see the raw elbow estimate to
            # tell whether that's the floor genuinely binding (elbow wants
            # to go lower) or just where convergence actually happens.
            if _raw_elbow_median < profile["min_iters_floor"]:
                print(f"[EarlyStop] elbow diag: raw median elbow={_raw_elbow_median} "
                      f"< floor={profile['min_iters_floor']} - floor is binding, "
                      f"elbows were {sorted(elbows)}")
            else:
                print(f"[EarlyStop] elbow diag: raw median elbow={_raw_elbow_median} "
                      f">= floor={profile['min_iters_floor']} - floor not binding, "
                      f"elbows were {sorted(elbows)}")
        else:
            new_min_iters = self.min_iters

        # Re-anchor target_iter_frac to the [min_iters, max_iters] range so that
        # the target iteration is always above min_iters. Without this, if
        # target_iter_frac * max_iters < min_iters (e.g. 0.60*100=60 < min_iters=70)
        # suggest_loss_eps can never satisfy the condition and always falls back.
        if max_iters > new_min_iters:
            _abs_target = new_min_iters + (max_iters - new_min_iters) * profile["target_iter_frac"]
            _eff_frac = _abs_target / max_iters
        else:
            _eff_frac = profile["target_iter_frac"]

        # The fit MUST use the criterion the loop is actually running.
        # suggest_loss_eps searches for the smallest eps whose SIMULATED fire
        # behaviour hits the rate/speed/quality targets, so simulating the
        # consecutive rule while the loop runs the window rule would tune a
        # threshold for a criterion that never executes - and periodic
        # retuning is load-bearing here (frozen thresholds gave ATE
        # 3.84/8.70/15.17 on identical code).
        new_loss_eps = suggest_loss_eps(
            frames, new_min_iters, self.patience, max_iters,
            target_iter_frac=_eff_frac,
            quality_threshold=profile["quality_threshold"],
            criterion=self.criterion, window=self.window,
            window_slack=self.window_slack,
        )

        # Diagnostic: compute_pose_tail_mass returns None when its internal
        # tail_masses list ends up empty (every frame skipped for having
        # <4 iters, no non-None pose values, or total pose mass <1e-12).
        # Report the raw pose_delta_norm distribution directly so we can
        # tell which of those is actually happening, instead of guessing.
        _n_frames = len(frames)
        _n_short = sum(1 for fd in frames if len(fd["signals"]) < 4)
        _all_pose_vals = [s[2] for fd in frames for s in fd["signals"] if s[2] is not None]
        if _all_pose_vals:
            print(f"[EarlyStop] pose diag: {_n_frames} probe frames ({_n_short} too short), "
                  f"{len(_all_pose_vals)} pose samples: "
                  f"min={min(_all_pose_vals):.2e} max={max(_all_pose_vals):.2e} "
                  f"mean={sum(_all_pose_vals)/len(_all_pose_vals):.2e} "
                  f"sum={sum(_all_pose_vals):.2e}")
        else:
            print(f"[EarlyStop] pose diag: {_n_frames} probe frames ({_n_short} too short), "
                  f"NO pose values found (all None)")

        tail_mass = compute_pose_tail_mass(frames, profile["quality_threshold"])
        if tail_mass is not None and tail_mass > _POSE_TAIL_MASS_THRESHOLD:
            new_pose_eps = 0.0
        else:
            new_pose_eps = suggest_pose_eps(
                frames, new_min_iters, max_iters,
                quality_threshold=profile["quality_threshold"],
            )

        # ── stabilise the fitted loss_eps ─────────────────────────────────────
        #
        # The stabilisers anchor on _loss_eps_base, NOT on self.loss_eps.
        # self.loss_eps is the value check() uses and already carries
        # retune_loss_eps_scale; anchoring on it would make the scale COMPOUND
        # through the clamp at every retune - measured, x1.6 reached 6.40e-03
        # after four retunes instead of 1.6x the unscaled 2.68e-03, i.e. an
        # effective 2.4x. A dial whose meaning depends on how many times it has
        # fired is not a dial. Keeping the pre-scale trajectory separate makes
        # the scale exactly a multiplier on the unscaled schedule.
        _raw_fit = new_loss_eps
        _notes = []
        _base = self._loss_eps_base
        if self.retune_reject_fallback and abs(new_loss_eps - 1e-4) < 1e-12:
            # Exactly 1e-4 is off suggest_loss_eps's search grid, so it can only
            # be the give-up return. Keep what we have.
            _notes.append(f"fit returned the 1e-4 fallback, keeping {_base:.2e}")
            new_loss_eps = _base
        elif self.retune_max_change > 1.0 and self._retune_count > 0:
            _lo = _base / self.retune_max_change
            _hi = _base * self.retune_max_change
            _clamped = min(max(new_loss_eps, _lo), _hi)
            if _clamped != new_loss_eps:
                _notes.append(f"clamped from {new_loss_eps:.2e} "
                              f"(max {self.retune_max_change:.2f}x per retune)")
                new_loss_eps = _clamped
        if 0.0 < self.retune_smooth < 1.0 and self._retune_count > 0:
            _smoothed = ((1.0 - self.retune_smooth) * _base
                         + self.retune_smooth * new_loss_eps)
            if _smoothed != new_loss_eps:
                _notes.append(f"smoothed from {new_loss_eps:.2e} "
                              f"(w={self.retune_smooth:.2f})")
                new_loss_eps = _smoothed

        self._loss_eps_base = new_loss_eps
        if self.retune_loss_eps_scale != 1.0:
            _notes.append(f"scaled x{self.retune_loss_eps_scale:.2f} "
                          f"from {new_loss_eps:.2e}")
            new_loss_eps *= self.retune_loss_eps_scale

        old_loss, old_pose, old_min = self.loss_eps, self.pose_eps, self.min_iters
        self.loss_eps  = new_loss_eps
        self.pose_eps  = new_pose_eps
        self.min_iters = new_min_iters

        # reset counters so new thresholds take effect cleanly
        self._plateau      = 0
        self._pose_plateau = 0
        self._prev         = None
        self._loss_hist    = []
        self._pose_hist    = []
        self._retune_count += 1

        # Always show tail_mass, not just when it disables pose_eps - the
        # threshold (0.60) is what decides pose_eps=0 vs suggest_pose_eps(),
        # so seeing the number confirms *why* either way, not just the
        # outcome. Was previously hidden whenever pose_eps came out nonzero,
        # making it impossible to tell "comfortably passed" from "barely".
        tail_note = f"  tail_mass={tail_mass:.2f}" if tail_mass is not None else "  tail_mass=n/a"
        tail_note += f" (threshold={_POSE_TAIL_MASS_THRESHOLD:.2f})"
        # Always print the RAW fit alongside the applied value. Without it the
        # stabilised runs and the unstabilised ones cannot be compared - the
        # log would only show what was used, never what the estimator wanted,
        # and profiling/retune_report.py reads this line.
        if _notes:
            tail_note += f"  [raw fit {_raw_fit:.2e}: " + "; ".join(_notes) + "]"
        print(f"[EarlyStop] Retune #{self._retune_count} @ frame {self.total_frames}: "
              f"loss_eps {old_loss:.2e}→{new_loss_eps:.2e}  "
              f"pose_eps {old_pose:.2e}→{new_pose_eps:.2e}  "
              f"min_iters {old_min}→{new_min_iters}{tail_note}")

    # ── reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        self.flush()
        if not self.enabled:
            return "Early stopping: disabled"
        pct = 100 * self.fires / self.total_frames if self.total_frames else 0
        avg = self._fire_iter_sum // self.fires if self.fires else 0
        s = (f"Early stopping fired: {self.fires}/{self.total_frames} frames "
             f"({pct:.1f}%), avg at iteration {avg}")
        if self._retune_count > 0:
            s += f"  [retuned {self._retune_count}×]"
        return s
