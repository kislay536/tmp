import math


def autoscale_iteration_bounds(cfg: dict, runtime_ceiling: int):
    """Scale configured mapping bounds to the caller's runtime ceiling.

    MonoGS's asynchronous mapper uses a ten-iteration keyframe ceiling while
    its inline/single-process mapper uses ``mapping_itr_num`` (normally 150).
    The configured bounds describe a *fraction* of the ceiling they were
    authored for; this helper preserves that fraction when the execution mode
    changes.  Scene calibration still fits novelty references only.

    ``bounds_reference_iters`` can pin the source ceiling explicitly.  When it
    is omitted, the configured ``max_iters`` is the reference, which is the
    natural interpretation for the existing 3..10 MonoGS configurations.
    Set ``auto_scale_bounds: False`` for a deliberately absolute budget.

    Returns ``(scaled_cfg, details)`` without mutating ``cfg``.  ``details`` is
    suitable for a startup log and makes the selected bounds testable without
    importing a model or CUDA.
    """
    runtime_ceiling = int(runtime_ceiling)
    if runtime_ceiling <= 0:
        raise ValueError("runtime mapping iteration ceiling must be positive")

    out = dict(cfg or {})
    old_min = int(out.get("min_iters", 10))
    old_max = int(out.get("max_iters", 60))
    if old_min <= 0 or old_max <= 0 or old_min > old_max:
        raise ValueError(
            f"invalid adaptive-mapping bounds {old_min}..{old_max}"
        )

    reference = int(out.get("bounds_reference_iters", old_max))
    if reference <= 0:
        raise ValueError("bounds_reference_iters must be positive")

    active = bool(out.get("enabled", False))
    automatic = bool(out.get("auto_scale_bounds", True))
    new_min, new_max = old_min, old_max
    if active and automatic and runtime_ceiling != reference:
        scale = runtime_ceiling / reference
        new_min = min(runtime_ceiling, max(1, int(round(old_min * scale))))
        new_max = min(runtime_ceiling, max(new_min, int(round(old_max * scale))))
        out["min_iters"] = new_min
        out["max_iters"] = new_max

    details = {
        "enabled": active,
        "automatic": automatic,
        "reference_ceiling": reference,
        "runtime_ceiling": runtime_ceiling,
        "configured_bounds": (old_min, old_max),
        "selected_bounds": (new_min, new_max),
        "scaled": (new_min, new_max) != (old_min, old_max),
    }
    return out, details


class AdaptiveMapper:
    """Decides per-frame mapping iteration budget based on scene novelty signals.

    Signals (normalised to [0, 1]):
        unseen_ratio    – fraction of pixels not covered by Gaussians (opacity < sil_thres)
        n_new_pts_ratio – newly seeded Gaussians / n_new_pts_ref
        depth_error     – mean |rendered - GT depth| from iter 0, normalised by depth_error_ref
        color_error     – L1 color loss from iter 0, normalised by color_error_ref

    Budget = min_iters + novelty * (max_iters - min_iters)
    where novelty = weighted sum of provided signals, normalised by sum of their weights.

    Weights for unused signals should be set to 0.0 in config so they don't dilute novelty.
    When disabled (enabled=False), always returns max_iters.
    """

    def __init__(self, cfg: dict):
        self.enabled        = bool( cfg.get("enabled",         False))
        self.min_iters      = int(  cfg.get("min_iters",        10))
        self.max_iters      = int(  cfg.get("max_iters",        60))
        self.w_unseen       = float(cfg.get("w_unseen",         0.40))
        self.w_new_pts      = float(cfg.get("w_new_pts",        0.30))
        self.w_depth        = float(cfg.get("w_depth",          0.20))
        self.w_color        = float(cfg.get("w_color",          0.10))
        self.depth_ref      = float(cfg.get("depth_error_ref",  0.05))
        self.color_ref      = float(cfg.get("color_error_ref",  0.05))
        self.n_new_pts_ref  = int(  cfg.get("n_new_pts_ref",    300))

        # Render-workload capping (optional, independent of iteration budget):
        # compute_budget() only ever changed how many mapping iterations run,
        # never what each iteration costs - every iteration still renders the
        # full keyframe window + random viewpoints regardless of novelty. At
        # low novelty, capping *what* each iteration renders (not just how
        # many iterations) reduces per-iteration cost directly instead of
        # only iteration count, which the earlier iteration-only budget
        # showed had a low ceiling on this workload (see exp_adaptive_mapping
        # notes - mapping's total iteration share is small, so cutting
        # iteration count alone bought only ~5% wall time).
        self.cap_render_workload = bool(cfg.get("cap_render_workload", False))
        self.min_keyframe_frac   = float(cfg.get("min_keyframe_frac",       0.5))
        self.max_keyframe_frac   = float(cfg.get("max_keyframe_frac",       1.0))
        self.min_random_viewpoints = int(cfg.get("min_random_viewpoints",     1))
        self.max_random_viewpoints = int(cfg.get("max_random_viewpoints",     2))
        self._last_novelty = 1.0  # full workload until a real novelty exists

        print(f"[AdaptiveMapper] constructed (enabled={self.enabled}, "
              f"min_iters={self.min_iters}, max_iters={self.max_iters}, "
              f"cap_render_workload={self.cap_render_workload})", flush=True)

        self._total_frames = 0
        self._iter_sum     = 0
        self._logged_first_call = False
        self._render_cap_frames = 0
        self._kf_rendered_sum   = 0
        self._kf_full_sum       = 0

        # THE RAW ERROR DISTRIBUTION, so depth_error_ref and color_error_ref
        # can be FITTED ON A SCENE instead of carried from the one they were
        # measured on.
        #
        # WHY THIS WAS NEEDED. The shipped 0.09/0.18 come from a real
        # 137-keyframe fr1_desk run (depth p90 0.095, color p90 0.189), and
        # nothing has ever recorded the same quantity anywhere else - only the
        # summary sums were kept. On Replica that showed up as a budget of
        # 4.1/10 against TUM's 7.5-8.0: those scenes render at PSNR 39-44, so
        # their errors sit far below a TUM-fitted denominator, novelty = error
        # / ref collapses, and the mechanism applies a near-constant 50-60%
        # cut rather than adapting to anything.
        #
        # HOST-SIDE AND FREE. Both values are already floats on the host by the
        # time they reach compute_budget - it does float() on them - so this is
        # two list appends per keyframe, at most a few hundred per run.
        self._depth_errs: list[float] = []
        self._color_errs: list[float] = []

        # IN-RUN SELF-CALIBRATION, the same shape as adaptive_diag's
        # calibration_frames: measure for N keyframes, install, freeze.
        #
        # WHY NOT A CALIBRATION RUN. Fitting from a prefix needs ~30 keyframes
        # for a p90, and keyframes arrive once per 5 to 31 frames depending on
        # the sequence - so a prefix long enough to fit is 150-200 frames, 30-40%
        # of a 500-frame run. That is a second experiment, not a tuner.
        #
        # THIS COSTS NOTHING EXTRA. During calibration the budget is max_iters,
        # which is exactly what the arm does with adaptive mapping OFF - so the
        # calibration window is baseline behaviour, not overhead. The only thing
        # forgone is the saving that would have been made during it, and that
        # saving measured zero anyway.
        #
        # AND IT DEGRADES SAFELY. A sequence that never reaches the sample count
        # never installs a reference and runs at max_iters throughout, which is
        # the base arm. fr2_xyz has 16 keyframes in 500 frames and will do
        # exactly that rather than fit a reference from six samples.
        self.calib_kf = int(cfg.get("calibration_keyframes", 0))
        # THE FIRST KEYFRAMES ARE DISCARDED, NOT USED. They are mapped against a
        # map that barely exists, so their errors are the largest of the run and
        # are not what a reference should describe. p90 is a tail statistic, so
        # including them fits a reference that is too LARGE - the direction that
        # already collapses novelty on Replica. Measured on a synthetic prefix
        # with a realistic transient: including them gave a colour p90 of 0.155
        # against a true 0.055.
        self.calib_skip = int(cfg.get("calibration_skip", 8))
        self._calibrated = (self.calib_kf <= 0)

        # IN-RUN FIT OF n_new_pts_ref, for models that supply the raw count.
        #
        # WHY THIS EXISTS. The calibration above fits the depth and colour
        # references from depth_error/color_error, and Gaussian-SLAM never
        # supplies those - mapper.py passes unseen_ratio and n_new_pts_ratio
        # only - so enabling it there holds the budget at max_iters forever.
        # The signal GSLAM does drive novelty from is n_new_pts / n_new_pts_ref,
        # and that reference is the one value that does not transfer between
        # scenes: TUM ships 5000, Replica needed 40000, and the class default
        # of 300 saturates the ratio at 1.0 on any scene that adds more.
        # ScanNet had no adaptive_mapping block at all, ran on those defaults
        # (budget 10-60 of a stock 100) and lost tracking at frame ~1000.
        #
        # SAME SHAPE AS THE DEPTH/COLOUR CALIBRATION: measure for N frames at
        # max_iters (which is what the arm does with adaptive mapping off, so
        # the window costs nothing extra), fit the p90 of the RAW count, freeze.
        # The raw count, not the ratio: the caller clamps the ratio at 1.0
        # before it arrives, and the p90 of a clamped series cannot say the
        # reference is too small.
        #
        # OFF BY DEFAULT (0), so no existing cell changes. If the caller never
        # supplies n_new_pts the calibration simply never starts and the normal
        # budget applies, rather than holding max_iters for the whole run.
        self.calib_np = int(cfg.get("calibrate_new_pts_frames", 0))
        self.calib_np_skip = int(cfg.get("calibrate_new_pts_skip", 5))
        self.calib_np_scale = float(cfg.get("calibrate_new_pts_scale", 1.0))
        if not math.isfinite(self.calib_np_scale) or self.calib_np_scale <= 0.0:
            raise ValueError(
                "calibrate_new_pts_scale must be finite and greater than zero"
            )
        self._np_calibrated = (self.calib_np <= 0)
        self._np_raw: list[float] = []
        self._np_fitted = False

    def compute_budget(self, unseen_ratio=None, n_new_pts_ratio=None,
                       depth_error=None, color_error=None,
                       n_new_pts=None) -> int:
        """Return the iteration budget for the current frame.

        Pass None for signals that are not available for this model — they are
        excluded from the weighted sum and do not dilute novelty.
        Always call once per mapped frame so summary() stats stay accurate.

        n_new_pts is the RAW number of newly seeded Gaussians (not the ratio).
        Optional; only needed for calibrate_new_pts_frames.
        """
        if not self._logged_first_call:
            # Fires regardless of enabled/disabled - confirms the calling
            # code's gate (self.initialized, is_new_submap, etc.) was
            # actually reached, which a conditional "budget shrunk" print
            # alone can't tell you (silence there is ambiguous between
            # "never called" and "always got the full budget").
            print(f"[AdaptiveMapper] compute_budget() reached for the first time "
                  f"(enabled={self.enabled}, min_iters={self.min_iters}, "
                  f"max_iters={self.max_iters})", flush=True)
            self._logged_first_call = True

        if not self.enabled:
            return self.max_iters

        # WAS STILL CALIBRATING AT THE START OF THIS CALL - captured before
        # either window's install below can flip it, so the call that
        # completes a window still gets max_iters, the same as every call
        # before it. Without this, the completing call would compute and
        # return a REAL budget from a reference that was only just installed
        # a few lines above it in the same call - one call earlier than every
        # other call in the window, and a one-off inconsistency the window's
        # own sample count doesn't account for.
        _was_calibrated = self._calibrated
        _was_np_calibrated = self._np_calibrated

        # RAW NEW-POINT COUNT, recorded on every call after the enabled check
        # and before anything can return, so the distribution is the one the
        # whole run saw (summary() reports it next to the reference).
        #
        # DOES NOT RETURN HERE, even while still fitting n_new_pts_ref - see
        # BUG note below. It only installs the reference when its own window
        # completes; whether the frame's budget is still full comes later,
        # once the depth/colour window has had an equal chance to run.
        if n_new_pts is not None:
            self._np_raw.append(float(n_new_pts))
            if (not self._np_calibrated
                    and len(self._np_raw) >= self.calib_np_skip + self.calib_np):
                self._install_new_pts_ref()

        novelty = 0.0
        total_w = 0.0

        if unseen_ratio is not None:
            s = min(max(float(unseen_ratio), 0.0), 1.0)
            novelty += self.w_unseen * s
            total_w += self.w_unseen

        if n_new_pts_ratio is not None:
            s = min(max(float(n_new_pts_ratio), 0.0), 1.0)
            novelty += self.w_new_pts * s
            total_w += self.w_new_pts

        if depth_error is not None:
            # RECORDED BEFORE THE CLAMP. min(..., 1.0) is what the budget
            # needs, but a fit needs the raw value - every keyframe whose
            # error exceeds the reference would otherwise be indistinguishable
            # from one exactly at it, and the p90 of a clamped series cannot
            # tell you the reference is too SMALL.
            self._depth_errs.append(float(depth_error))
            s = min(float(depth_error) / (self.depth_ref + 1e-8), 1.0)
            novelty += self.w_depth * s
            total_w += self.w_depth

        if color_error is not None:
            self._color_errs.append(float(color_error))
            s = min(float(color_error) / (self.color_ref + 1e-8), 1.0)
            novelty += self.w_color * s
            total_w += self.w_color

        # STILL CALIBRATING: run at full budget and install nothing yet.
        #
        # RETURNS BEFORE THE BUDGET IS COMPUTED, deliberately. The novelty above
        # was evaluated against references that are known-wrong for this scene -
        # that is the whole reason to calibrate - so acting on it during the
        # window would apply the very cut being measured, and the errors that
        # cut produced would then feed the fit. Full budget keeps the observed
        # distribution the one the base arm actually sees.
        if not self._calibrated:
            n = max(len(self._depth_errs), len(self._color_errs))
            if n >= self.calib_skip + self.calib_kf:
                self._install_calibrated_refs()

        # BOTH WINDOWS GATE THE BUDGET, not just this one - and each runs on
        # its OWN clock, tracked independently above, not one after the
        # other. A caller that sets both calibration_keyframes AND
        # calibrate_new_pts_frames (first exercised by SplaTAM - GSLAM only
        # ever used the new-points window, MonoGS only ever used this one)
        # would otherwise have depth_error/color_error accumulation BLOCKED
        # from ever starting until the new-points window finished first: the
        # two windows would serialise into one roughly twice as long, and
        # whichever calibrates second would fit over a later, less
        # representative slice of the run than a solo window would have. A
        # caller that only ever supplies one of the two signals is
        # unaffected - the other's own _calibrated flag is already True from
        # construction (calib_kf or calib_np <= 0), so this reduces to the
        # single check it always was.
        #
        # THE CAPTURED "WAS" FLAGS, not the live ones - see the comment where
        # they're captured above.
        #
        # THE NEW-POINTS HALF IS ALSO GATED ON n_new_pts NOT BEING None THIS
        # CALL, same as the original single-window check was - a caller that
        # requests calibrate_new_pts_frames but never actually supplies the
        # raw count (wrong signal for this model, or a config copied from one
        # that does) must not be held at max_iters forever waiting for a
        # window that can never complete. The depth/colour half has no such
        # escape and never did: calibration_keyframes > 0 on a caller that
        # never supplies depth/colour error IS held at max_iters forever -
        # the exact failure this file's own history records for GSLAM, which
        # is why GSLAM uses calibrate_new_pts_frames instead.
        if (not _was_calibrated) or (not _was_np_calibrated and n_new_pts is not None):
            self._total_frames += 1
            self._iter_sum += self.max_iters
            self._last_novelty = 1.0
            return self.max_iters

        if total_w > 0:
            novelty /= total_w
        novelty = min(max(novelty, 0.0), 1.0)
        self._last_novelty = novelty

        budget = int(round(self.min_iters + novelty * (self.max_iters - self.min_iters)))
        budget = min(max(budget, self.min_iters), self.max_iters)

        self._total_frames += 1
        self._iter_sum     += budget
        return budget

    def compute_render_caps(self, window_size: int) -> tuple:
        """Return (max_keyframes, max_random_viewpoints) to render per mapping
        iteration, using the novelty from the most recent compute_budget()
        call - call compute_budget() first each frame, this doesn't
        recompute novelty from scratch.

        Unlike compute_budget() (iteration *count*), this caps what each
        iteration renders - low novelty means fewer keyframes/random
        viewpoints processed per iteration, not just fewer iterations.
        Returns (window_size, max_random_viewpoints) - i.e. no reduction -
        when disabled or cap_render_workload=False, so callers that don't
        check the flag still get correct, unreduced behavior.
        """
        if not self.enabled or not self.cap_render_workload:
            return window_size, self.max_random_viewpoints

        novelty = self._last_novelty
        max_kf = int(round(
            window_size * (self.min_keyframe_frac
                            + novelty * (self.max_keyframe_frac - self.min_keyframe_frac))
        ))
        max_kf = min(max(max_kf, 1), window_size)
        max_rv = int(round(
            self.min_random_viewpoints
            + novelty * (self.max_random_viewpoints - self.min_random_viewpoints)
        ))
        max_rv = min(max(max_rv, 0), self.max_random_viewpoints)

        self._render_cap_frames += 1
        self._kf_rendered_sum   += max_kf
        self._kf_full_sum       += window_size
        return max_kf, max_rv

    def summary(self) -> str:
        if not self.enabled:
            return "Adaptive mapping: disabled"
        if self._total_frames == 0:
            return "Adaptive mapping: enabled, 0 frames processed"
        avg     = self._iter_sum / self._total_frames
        speedup = self.max_iters / avg if avg > 0 else 1.0
        s = (f"Adaptive mapping: {self._total_frames} frames, "
             f"avg {avg:.1f}/{self.max_iters} iters ({speedup:.2f}x speedup)")
        if self._render_cap_frames > 0:
            kf_avg = self._kf_rendered_sum / self._render_cap_frames
            kf_full_avg = self._kf_full_sum / self._render_cap_frames
            s += (f"; render cap: {self._render_cap_frames} calls, "
                  f"avg {kf_avg:.1f}/{kf_full_avg:.1f} keyframes rendered/iter")
        s += self.reference_fit()
        s += self.new_pts_fit()
        return s

    def _install_calibrated_refs(self) -> None:
        """Fit both references from this scene's own errors and freeze them.

        CALIBRATE THEN FREEZE, not a running estimate. This record has five
        instances of per-frame signals failing to predict trajectory-level
        outcomes, and the adaptive-diagonal criterion specifically was proven
        harmless in shadow and then diverged when installed live. A reference
        that keeps moving makes every frame's budget depend on how the previous
        frames happened to render, which is the same failure wearing different
        clothes. One fit, installed once, printed once.
        """
        d = self._depth_errs[self.calib_skip:]
        c = self._color_errs[self.calib_skip:]
        old_d, old_c = self.depth_ref, self.color_ref
        if d:
            self.depth_ref = max(self._pct(d, 90), 1e-6)
        if c:
            self.color_ref = max(self._pct(c, 90), 1e-6)
        self._calibrated = True
        print(f"[AdaptiveMapper] calibrated on {len(d) or len(c)} keyframes "
              f"(first {self.calib_skip} discarded as map warm-up): "
              f"depth_error_ref {old_d:g} -> {self.depth_ref:.4f} "
              f"({self.depth_ref / old_d:.2f}x), "
              f"color_error_ref {old_c:g} -> {self.color_ref:.4f} "
              f"({self.color_ref / old_c:.2f}x)", flush=True)

    def _install_new_pts_ref(self) -> None:
        """Fit n_new_pts_ref to the p90 of this scene's own raw new-point counts.

        p90, the same statistic the depth/colour references use: the reference
        is a novelty DENOMINATOR that saturates at 1.0, and p90 leaves the top
        decile saturated - those frames genuinely want the full budget. The
        first calibrate_new_pts_skip frames are discarded: they are mapped
        against a map that barely exists and seed far more than the run's
        steady state, which would fit a reference that is too LARGE.
        Calibrate then freeze, never a running estimate.
        """
        xs = self._np_raw[self.calib_np_skip:]
        old = self.n_new_pts_ref
        p50, p90 = self._pct(xs, 50), self._pct(xs, 90)
        scaled_p90 = p90 * self.calib_np_scale
        self.n_new_pts_ref = max(int(round(scaled_p90)), 1)
        self._np_calibrated = True
        self._np_fitted = True
        print(f"[AdaptiveMapper] calibrated n_new_pts_ref on {len(xs)} frames "
              f"(first {self.calib_np_skip} discarded as map warm-up): "
              f"{old} -> {self.n_new_pts_ref} ({self.n_new_pts_ref / max(old, 1):.2f}x); "
              f"new points p50 {p50:.0f} p90 {p90:.0f} "
              f"scale {self.calib_np_scale:g} scaled_p90 {scaled_p90:.0f} "
              f"max {max(xs):.0f}", flush=True)

    def new_pts_fit(self) -> str:
        """The raw new-point distribution over the whole run, against the
        reference in force - the actionable ratio, as reference_fit() does for
        depth/colour. fit/current far above 1 means the reference is too small
        (novelty pinned at 1.0, budget pinned at max_iters); far below 1, too
        large (budget collapsed toward min_iters)."""
        if not self._np_raw:
            return ""
        p50, p90 = self._pct(self._np_raw, 50), self._pct(self._np_raw, 90)
        ref = self.n_new_pts_ref
        how = ("fitted in-run" if self._np_fitted else "configured")
        return (f"\n  NEW POINTS over {len(self._np_raw)} frames: p50 {p50:.0f} "
                f"p90 {p90:.0f} (ref {ref}, {how}, fit/current {p90 / max(ref, 1):.2f}x)")

    @staticmethod
    def _pct(xs, q):
        """Nearest-rank percentile. No numpy dependency for ~100 floats."""
        if not xs:
            return float("nan")
        ys = sorted(xs)
        k = min(len(ys) - 1, max(0, int(round(q / 100.0 * (len(ys) - 1)))))
        return ys[k]

    def reference_fit(self) -> str:
        """What depth_error_ref/color_error_ref SHOULD be for this scene.

        THE FIT IS THE p90, because that is how the shipped 0.09/0.18 were
        derived - a 137-keyframe fr1_desk run measured depth p90 0.095 and
        color p90 0.189. Using the same statistic keeps a re-fit on a new scene
        comparable with the values already in the configs rather than
        introducing a second convention.

        WHY p90 AND NOT THE MEAN. The reference is a novelty DENOMINATOR that
        saturates: error/ref is clamped at 1.0. Fitting the mean would put half
        the keyframes at saturation and throw away the mechanism's whole upper
        range. p90 leaves the top decile saturated, which is the intent - those
        frames genuinely want the full budget.

        THE RATIO IS PRINTED BECAUSE IT IS THE ACTIONABLE NUMBER. fit/current
        far below 1 means the current reference is too LOOSE for this scene:
        novelty collapses and the budget pins toward min_iters whether or not
        anything was novel. Far above 1 means too tight, novelty saturates, and
        the budget pins to max_iters - the failure the old 0.01/0.05 produced.
        """
        if not self._depth_errs and not self._color_errs:
            return ""
        # THE WARM-UP PROBES ARE REPORTED SEPARATELY, NOT DROPPED.
        #
        # The first keyframes are mapped against a map that barely exists, so
        # their errors are the highest of the run and are NOT what the
        # reference should describe. p90 is a tail statistic, so a short
        # calibration prefix made mostly of those probes fits a reference that
        # is too LARGE - which is the same direction that is already breaking
        # Replica, and would look like agreement rather than error.
        #
        # Both are printed rather than one silently chosen: the shipped
        # 0.09/0.18 were fitted over a whole 137-keyframe run with no warm-up
        # exclusion, so an excluded fit is not directly comparable with them.
        # If the two agree, the prefix was long enough and either can be used.
        # If they diverge, the run was too short - that is the signal to
        # lengthen it, not to pick the number you prefer.
        _WARM = 10
        out = []
        for label, xs, ref in (("depth", self._depth_errs, self.depth_ref),
                               ("color", self._color_errs, self.color_ref)):
            if not xs:
                continue
            p50, p90 = self._pct(xs, 50), self._pct(xs, 90)
            ratio = p90 / ref if ref > 0 else float("nan")
            seg = f"{label} p50 {p50:.4f} p90 {p90:.4f} " \
                  f"(ref {ref:g}, fit/current {ratio:.2f}x)"
            if len(xs) > _WARM + 5:
                p90w = self._pct(xs[_WARM:], 90)
                seg += f" [excl. first {_WARM}: p90 {p90w:.4f}]"
            out.append(seg)
        if not out:
            return ""
        return ("\n  REFERENCE FIT over " + str(len(self._depth_errs) or
                len(self._color_errs)) + " probes: " + "; ".join(out) +
                "\n  -> set depth_error_ref/color_error_ref to the p90 values "
                "above to fit this scene; a fit/current far from 1.00x means "
                "the shipped TUM-derived reference does not transfer here.")
