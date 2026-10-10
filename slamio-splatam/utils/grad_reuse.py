"""Render on some tracking iterations and reuse the gradient on the rest.

THE MEASUREMENT THAT JUSTIFIES THIS. utils/grad_staleness.py, run at the
SHIPPED operating point (PRESET=final, ~27 iterations/frame, WANDER ~3.2):

    offset   cos(stale)   |g0|/|g|
       1       0.9473       0.942
       2       0.5141       0.976
       4      -0.2658       1.086

One step stale, the pose gradient still points 0.947 in the same direction at
0.942 of the magnitude. Two steps stale it is 0.51, and by four it is
ANTI-PARALLEL. So period 2 is the only setting the data supports, and
`period` is capped accordingly unless explicitly overridden.

Those numbers are regime-specific and the regime matters more than anything
else here: the same probe read 0.7635 at one step when run at 200
iterations/frame with WANDER ~7. A gradient goes stale faster in a more
oscillatory optimiser, so a config that changes the step size or the stopping
rule invalidates the measurement above before it invalidates anything else.

WHAT IT SAVES. A skipped iteration skips the WHOLE pipeline - render, loss,
backward, and with them preprocess, binning and the sort - not one stage of
it. That is why this is worth more than tile masking (6-18% for 75% of tiles)
or binning reuse (null, the sort is ~6.7% of kernel time), both measured dead
on this branch.

THE TRAP THIS BRANCH ALREADY WALKED INTO ONCE. Sparse tile sampling cut
per-iteration cost and came out NET NEGATIVE per frame, because a slightly
worse gradient made the stopping rule run more iterations than the cheaper
ones saved. The same trade is live here and is sharper: the reused gradient is
at cos 0.947, not 1.0.

    PRICE THIS IN TRACKING SECONDS PER FRAME. NEVER ms/iter.

ms/iter will improve almost by construction - half the iterations do no
render. That number is not the result and must not be quoted as one.

AND THE STOPPING RULE SEES A REPEATED LOSS. A reuse iteration produces no new
loss, so the caller hands the stopping rule the previous one. An incumbent rule
needs a STRICTLY better loss to reset its patience, so a repeat reads as "no
improvement" and patience advances - stopping will tend to fire EARLIER. That
is a real interaction, not a bug to paper over, and it is the first thing to
look at if iterations/frame moves.

ADAPTIVE MODE (GRAD_REUSE_ADAPTIVE=1) replaces the fixed `cooldown` index with
a per-frame boundary that MOVES based on evidence, instead of a single global
guess. It exists because every attempt to find one fixed cooldown that works
across frames failed: GSLAM/fr1_desk frames run 84-200 iterations, and the
staleness probe found the one-step cosine does not decay monotonically with
absolute iteration number - it tracks how close THAT frame is to converging,
which a fixed index cannot see. One run even implicated the fixed index
itself: iteration 20 lines up with the preconditioner's DIAG regime switch
(`restarts N/N frames @it20`), and probing exactly there produced a spurious
sign flip unrelated to genuine staleness.

THE GATE CANNOT LOOK AT THE GRADIENT IT WOULD SKIP - computing it defeats the
whole point. The only signal available without paying for a render is
RETROSPECTIVE: compare the two most recent FRESH gradients (the pair that
straddles the reuse period) and use their agreement as evidence for whether
the NEXT candidate iteration is still safe to reuse.

THE SYNC COST IS THE BINDING CONSTRAINT, not accuracy. This file already notes
that reading the step norm on the host was measured at ~24% of tracking wall
time in an earlier attempt - a host sync on every candidate iteration would
plausibly cost more than reuse saves. So the cosine is computed ON DEVICE every
fresh iteration (near-free) and read back to the host only every `check_every`
fresh iterations, batching the sync the same way es_signals batches its own
drains (0.13 syncs/iter there). A trusted check EXTENDS the reuse boundary
further into the frame; a failed check FREEZES it where it is - a bad reading
(the it20 artefact, say) just ends reuse a little early for that one frame,
which is the safe failure direction, not a dangerous one.

A COOLDOWN CEILING IS NOT OPTIONAL, EVEN AT A STRICT trust_cos. Measured on
GSLAM/fr1_desk with freeze_m on: trust_cos=0.95 (grad_staleness.py's own
"viable raw" tier) with NO ceiling still cost -1.0 dB PSNR, -0.013 SSIM and
left 78% of frames running the full iteration budget (the boundary saturating
near frame length in most frames) versus the SAME config capped at
cooldown=70, which matched an unreused baseline on every metric - PSNR, SSIM,
LPIPS and ATE alike. A looser trust_cos=0.85 without a ceiling was far worse:
-3.7 dB PSNR. Direction staying correlated (what cosine measures) is
necessary but not sufficient evidence that continuing to reuse is safe - it
says nothing about whether the OPTIMIZER STATE reuse feeds (momentum, the
thing freeze_m targets) has stopped drifting. So `cooldown`, when set,
remains an absolute ceiling on the adaptive boundary rather than something to
retire once the threshold is tuned; the constructor warns if it is left at 0.

SHORT-HORIZON CALIBRATION is the escape hatch for trackers whose own stopping
rule ends a frame almost immediately after the preconditioner's phase boundary.
Forcing warmup=DIAG is degenerate there: SplaTAM stops around iteration 25-28
with DIAG=20, leaving only two to four reuse candidates.  When explicitly
enabled, the first no-reuse calibration frames therefore test whether the
earliest real stop leaves two complete adaptive-check horizons after DIAG.  If
not, the calibrated window is expressed as fractions of that stop, aligned to
the reuse period: one quarter fresh warmup and a final one-fifth fresh tail.
This gives the measured SplaTAM operating points 25 -> w6/c20 and 28 -> w6/c22.
The short window is deliberately FIXED: SplaTAM's measured two-step cosine is
0.5141, so the long-frame adaptive gate would freeze at its initial boundary
and make the supposedly enabled mechanism nearly inert.
"""

import os

import numpy as np
import torch


class GradReuse:
    """Decides which tracking iterations reuse the previous pose gradient."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.period = int(cfg.get("period", 2))
        # Iterations at the START of a frame that always render. The pose moves
        # fastest there - the step profile shows it0-4 at 1.13x the reference
        # against 0.08x after iteration 40 - so a stale gradient is worth least
        # exactly where the optimiser is moving most.
        self.warmup = int(cfg.get("warmup", 2))
        self.allow_long_period = bool(cfg.get("allow_long_period", False))
        # COOLDOWN: stop reusing from this iteration on. 0 disables it.
        #
        # THE COMPLEMENT OF WARMUP, AND IT FIXES A DIFFERENT THING.
        # Warmup protects the OPENING of a frame, where the pose moves
        # fastest, and measurably buys ATE - on MonoGS fr1_desk it took
        # ATE 2.113 -> 1.908 going from 6 to 16.
        #
        # It does nothing for ITERATION COUNT, and that was measured too:
        # warmup 6 and warmup 16 gave byte-identical stop counts, native
        # 14 / cap 577 out of 591 frames, against the baseline's 543 / 48.
        # The frames never converged in either.
        #
        # The reason is at the other end. MonoGS spends roughly 50 of its
        # ~70 iterations at steps below 0.10x the reference - not
        # converging but POLISHING - and reusing a gradient there steps on
        # information from when the pose was further away, so the step
        # never decays and a step-magnitude convergence test can never
        # fire. Under reuse that band sits at 0.35x instead of 0.10x.
        #
        # A FIXED INDEX, DELIBERATELY, AND IT IS THE CRUDE VERSION. The
        # principled gate is on the step norm itself - reuse while moving,
        # render while settling - but reading the step norm on the host is
        # a sync, and this branch measured that tax at ~24% of tracking
        # wall time. An index costs nothing and tests the same hypothesis;
        # if it works, the adaptive version is worth its sync budget.
        self.cooldown = int(cfg.get("cooldown", 0))
        # GRAD_REUSE_STEP_SCALE: damp the APPLIED step on a reuse iteration,
        # independent of whether the stale gradient feeds m/M (that is
        # no_accum/freeze_m's job, below). The staleness probe measured the
        # reused gradient at cos 0.947 to the true direction, not 1.0 - a
        # smaller step limits how far the pose travels on a direction known
        # to be somewhat wrong, before the next fresh gradient corrects it.
        # Applied to the FINAL, trust-region-clamped delta at the call site
        # (tracker.py/slam_frontend.py/splatam.py, after pose_pre.step()),
        # not inside the preconditioner itself - it never touches m, M or P.
        #
        # PAIR WITH freeze_m (or no_accum), NOT PLAIN ACCUMULATE-BOTH. If m
        # is reinforced by the FULL reused gradient every reuse iteration
        # while the applied step only moves the pose by step_scale of that,
        # m ends up representing more displacement than actually happened -
        # a mismatch that compounds every reuse iteration until the next
        # fresh render resets the ground truth. freeze_m sidesteps this
        # entirely: m is not reinforced by reuse at all, so a scaled-down
        # delta has nothing to be inconsistent with. Not validated - this is
        # a new, not-yet-measured lever, separate from every knob above.
        #
        # 1.0 (default) is a no-op: nothing changes unless explicitly set.
        self.step_scale = float(cfg.get("step_scale", 1.0))
        if self.step_scale <= 0.0:
            raise ValueError("grad reuse step_scale must be > 0")
        # GRAD_REUSE_NOACC=1. On a reuse iteration, step but do NOT fold the
        # repeated gradient into the preconditioner's moment estimates - it
        # is the same measurement counted twice, not a second measurement.
        # Separate flag so it is a clean A/B against plain reuse rather than
        # a silent change to it.
        self.no_accum = bool(cfg.get("no_accum", False))
        # GRAD_REUSE_FREEZE_M=1: freeze MOMENTUM only on a reuse
        # iteration, letting the second moment advance.
        #
        # The middle option between accumulating both (the default, which
        # keeps the step from decaying) and freezing both (no_accum, which
        # was catastrophic on SplaTAM at ATE 3.5 -> 10.8).
        #
        # THE TWO ARE NOT THE SAME KNOB, and it is not obvious from reading:
        # P is only rebuilt by refactor(), so whether M advanced inside
        # step() cannot affect THAT step - on the reuse iteration itself
        # no_accum and freeze_m return byte-identical vectors. They diverge
        # only at the next refactor. Over GSLAM's real refactor_every=10 the
        # divergence is ~11% per step, and freeze_m takes the SMALLER steps
        # in the polish band (8.3e-3 vs 9.4e-3 default vs 9.6e-3 no_accum),
        # which is the direction the energy gate needs.
        #
        # NUMBERS TO HIT, so this arm can fail honestly. GSLAM/fr1_desk
        # w6c40 against its baseline: it40+ band 0.06x -> 0.03x, WANDER
        # path/net 23.7 -> 16.3, stopped-frame median 124 -> ~117. If
        # none of those move, the residual is not in m and this line
        # closes.
        self.freeze_m = bool(cfg.get("freeze_m", False))
        if self.freeze_m and self.no_accum:
            raise ValueError(
                "grad_reuse freeze_m and no_accum are alternatives: "
                "no_accum already freezes m (and M with it), so setting "
                "both hides which one is being tested")

        # ADAPTIVE COOLDOWN. See the module docstring for the mechanism and
        # why the sync has to be batched. cooldown, if set, becomes an
        # ABSOLUTE CEILING the adaptive boundary may never cross - a safety
        # net, not the primary control - rather than being ignored.
        self.adaptive = bool(cfg.get("adaptive", False))
        self.trust_cos = float(cfg.get("trust_cos", 0.85))
        # OPTIONAL SECOND GATE, OFF BY DEFAULT. trust_cos alone answers "is
        # the DIRECTION still trustworthy" - it says nothing about whether
        # the OPTIMIZER STATE reuse feeds has stopped drifting in SCALE. The
        # staleness probes (utils/grad_staleness.py) measured direction and
        # magnitude as genuinely different failure shapes: cosine is
        # persistently noisy throughout a frame, while magnitude is mostly
        # fine with occasional sharp excursions (4-8x) scattered anywhere in
        # the range - a check that only looks at cos would never see those.
        # When enabled, a fresh-gradient pair must ALSO keep |g_now|/|g_prev|
        # inside [trust_mag_low, trust_mag_high] to extend the boundary; if
        # either bound is broken the check fails exactly like a failed
        # cosine check does (freeze, don't shrink - see note_fresh_grad).
        self.trust_mag_enabled = bool(cfg.get("trust_mag_enabled", False))
        self.trust_mag_low = float(cfg.get("trust_mag_low", 0.8))
        self.trust_mag_high = float(cfg.get("trust_mag_high", 1.2))
        if self.trust_mag_enabled and self.trust_mag_low >= self.trust_mag_high:
            raise ValueError(
                "grad reuse trust_mag_low (%g) must be < trust_mag_high (%g)"
                % (self.trust_mag_low, self.trust_mag_high))
        # LOCAL_TRUST, OFF BY DEFAULT: replaces the monotonic boundary with a
        # live flag. The DEFAULT mode (_boundary) only ever GROWS on a
        # trusted check and FREEZES (stops growing) on a failed one - it
        # never actually revokes reuse that a later failed check should have
        # caught, because everything already inside the boundary keeps
        # reusing regardless of what happens afterward. That is deliberate
        # there (a spurious one-off bad reading should not retroactively
        # undo otherwise-fine iterations), but it also means a PERSISTENT
        # late-frame breakdown in trust never actually turns reuse back off
        # once an early run of good checks has pushed the boundary past it.
        #
        # local_trust=True drops the boundary/ratchet entirely: should_reuse
        # consults ONLY the most recent check's outcome. A failure stops
        # reuse immediately (not just stops further extension), and a LATER
        # success can turn it back on - reuse can flicker on and off through
        # a frame instead of being a one-way ratchet. cooldown, if set,
        # still applies as an absolute ceiling either way.
        self.local_trust = bool(cfg.get("local_trust", False))
        # Optional per-frame circuit breaker. Once this many consecutive
        # trust checks reject reuse, render fresh and stop checking for the
        # rest of the frame. The latch resets at the next frame, so reuse can
        # refire from the start there. This removes repeated host syncs on a
        # frame that is persistently untrustworthy without ever authorizing
        # an extra reuse step; the only possible error is conservatively
        # missing a later recovery in the same frame.
        self.reject_patience = int(cfg.get("reject_patience", 0))
        if self.reject_patience < 0:
            raise ValueError("grad reuse reject_patience must be >= 0")
        # Optional synchronous rejection backoff. A failed local-trust check
        # already forbids reuse, so checking every subsequent fresh render
        # only repeats the blocking device-to-host read while the state is
        # known bad. Wait this many REAL (rendered) iterations between
        # recovery probes. Gradients remain fresh throughout; the first
        # successful probe re-enables reuse in the same frame. Unlike the
        # reject_patience latch, this never gives up on recovery.
        self.reject_backoff = int(cfg.get("reject_backoff", 0))
        if self.reject_backoff < 0:
            raise ValueError("grad reuse reject_backoff must be >= 0")
        # Units: number of FRESH gradients between host reads, not real
        # iterations. With period=2 (a fresh gradient every 2 iterations),
        # check_every=4 is one sync per 8 real iterations - the same cadence
        # es_signals already pays for elsewhere, so this does not open a new
        # category of cost.
        self.check_every = int(cfg.get("check_every", 4))
        # ASYNC_CHECK, OFF BY DEFAULT: hides the host-sync latency instead of
        # paying it inline. The DEFAULT path blocks on .item()/.tolist() the
        # moment a check window closes, stalling the GPU until the value
        # lands - measured to cost more than check_every=1's tighter reuse
        # decisions were worth on at least one real run (SLAM-only time
        # WORSE than reuse disabled entirely: 667.5s vs 651.6s no-reuse).
        #
        # async_check=True issues a non-blocking copy to pinned host memory
        # plus a CUDA event instead, and only CONSUMES the result once it has
        # actually landed (_poll_pending_check(), called opportunistically
        # from should_reuse()/note_fresh_grad() - never blocks). While a copy
        # is pending, reuse FAILS CLOSED to a fresh render. A landed decision
        # is eligible to drive reuse only when no newer gradient evidence has
        # accumulated behind it; otherwise that backlog is checked first.
        # Decision latency may widen, but stale trust is never speculated on
        # and evidence is not lost. See note_fresh_grad() and
        # _poll_pending_check() for the mechanism.
        #
        # OFF BY DEFAULT so no existing cell's numbers move. Conservative
        # fallback renders can replace candidate reuse iterations when a
        # copy is late, so this remains its own tagged arm rather than a
        # silent drop-in replacement for the synchronous path.
        self.async_check = bool(cfg.get("async_check", False))
        # Exact deferred readback: issue the same pinned D2H copy as the async
        # path, but consume it (waiting if necessary) at the NEXT
        # should_reuse() decision. That preserves the synchronous decision
        # sequence exactly while allowing the copy to overlap refactor,
        # stopping-signal bookkeeping, and other between-iteration work.
        self.deferred_check = bool(cfg.get("deferred_check", False))
        # BATCHED CHECK, OFF BY DEFAULT. The cosine remains a GPU tensor and
        # is copied to the host as an extra column of SplaTAM's existing
        # early-stop signal drain. A trusted drain grants a bounded number of
        # future period-2 reuse slots (a lease); credits are replaced at every
        # drain, never accumulated. This removes the standalone .item() sync
        # while keeping the result exact and using it only prospectively.
        self.batched_check = bool(cfg.get("batched_check", False))
        self.trust_lease = int(cfg.get("trust_lease", 2))
        self.batched_signal_cols = 3 if self.trust_mag_enabled else 1
        if self.trust_lease < 1:
            raise ValueError("grad reuse trust_lease must be >= 1")
        if self.async_check and self.deferred_check:
            raise ValueError(
                "grad reuse async_check and deferred_check are alternatives")
        if self.batched_check and not self.adaptive:
            raise ValueError(
                "grad reuse batched_check requires adaptive=True")
        if self.batched_check and not self.local_trust:
            raise ValueError(
                "grad reuse batched_check requires local_trust=True")
        if self.batched_check and (self.async_check or self.deferred_check):
            raise ValueError(
                "grad reuse batched_check replaces async/deferred readback")
        if self.batched_check and self.reject_backoff:
            raise ValueError(
                "grad reuse batched_check does not use reject_backoff: "
                "failed drains already revoke the lease without another sync")
        if self.reject_backoff and not self.adaptive:
            raise ValueError(
                "grad reuse reject_backoff requires adaptive=True")
        if self.reject_backoff and not self.local_trust:
            raise ValueError(
                "grad reuse reject_backoff requires local_trust=True")
        if self.reject_backoff and self.async_check:
            # Async results are consumed later than their source iteration;
            # defining a real-iteration retry deadline from that stale index
            # would make the intended probe cadence ambiguous.
            raise ValueError(
                "grad reuse reject_backoff requires synchronous checks")
        # Optional run-in calibration. While collecting these frame-level
        # stopping points, reuse is completely disabled. The resulting
        # cooldown is derived from their median; see note_frame_stop().
        self.calibration_frames = int(cfg.get("calibration_frames", 0))
        # Independent no-reuse bootstrap.  Unlike calibration_frames this
        # does not inspect frame stopping reasons and does not derive or
        # mutate warmup/cooldown.  It exists for another controller (for
        # example the automatic stopping-rule selector) to collect an
        # uncontaminated prefix while every tracking iteration is rendered.
        self.hold_frames = int(cfg.get("hold_frames", 0))
        self.calibration_margin = int(cfg.get("calibration_margin", 15))
        self.calibration_round = int(cfg.get("calibration_round", 10))
        self.calibration_stat = str(cfg.get("calibration_stat", "median"))
        _excluded = cfg.get("calibration_exclude_reasons", ())
        if isinstance(_excluded, str):
            _excluded = [x.strip() for x in _excluded.split(",") if x.strip()]
        self.calibration_exclude_reasons = frozenset(
            str(x) for x in _excluded
        )
        # SHORT-HORIZON FALLBACK. Opt-in so the already measured MonoGS and
        # GSLAM calibration paths remain byte-for-byte unchanged.  The phase
        # boundary is normally DIAG.  If the earliest eligible stop leaves
        # fewer than `min_horizon` iterations after it, there is not enough
        # frame left for two adaptive trust decisions (2 * check_every *
        # period by default), so select a fixed, period-aligned window from
        # the stop itself instead.
        self.calibration_short_horizon = bool(
            cfg.get("calibration_short_horizon", False)
        )
        self.calibration_phase_boundary = int(
            cfg.get("calibration_phase_boundary", self.warmup)
        )
        self.calibration_min_horizon = int(cfg.get(
            "calibration_min_horizon", 2 * self.check_every * self.period
        ))
        self.calibration_short_warmup_fraction = float(
            cfg.get("calibration_short_warmup_fraction", 0.25)
        )
        self.calibration_short_cooldown_fraction = float(
            cfg.get("calibration_short_cooldown_fraction", 0.80)
        )
        # Three period-2 spans protect iterations 0-5, the high-motion band
        # measured on SplaTAM, even if an unusually early stop would make one
        # quarter of the frame smaller than that.
        self.calibration_short_min_warmup_periods = int(
            cfg.get("calibration_short_min_warmup_periods", 3)
        )
        if self.adaptive:
            if not (-1.0 <= self.trust_cos <= 1.0):
                raise ValueError("grad_reuse trust_cos must be in [-1, 1]")
            if self.check_every < 1:
                raise ValueError("grad_reuse check_every must be >= 1")
        if self.calibration_frames < 0:
            raise ValueError("grad reuse calibration_frames must be >= 0")
        if self.hold_frames < 0:
            raise ValueError("grad reuse hold_frames must be >= 0")
        if self.calibration_margin < 0:
            raise ValueError("grad reuse calibration_margin must be >= 0")
        if self.calibration_round < 1:
            raise ValueError("grad reuse calibration_round must be >= 1")
        if self.calibration_stat not in ("median", "min"):
            raise ValueError(
                "grad reuse calibration_stat must be 'median' or 'min'"
            )
        if self.calibration_short_horizon:
            if self.calibration_frames <= 0:
                raise ValueError(
                    "grad reuse short-horizon calibration requires "
                    "calibration_frames > 0"
                )
            if self.calibration_phase_boundary < 0:
                raise ValueError(
                    "grad reuse calibration_phase_boundary must be >= 0"
                )
            if self.calibration_min_horizon < 1:
                raise ValueError(
                    "grad reuse calibration_min_horizon must be >= 1"
                )
            if self.calibration_short_min_warmup_periods < 1:
                raise ValueError(
                    "grad reuse calibration_short_min_warmup_periods must "
                    "be >= 1"
                )
            if not (0.0 < self.calibration_short_warmup_fraction
                    < self.calibration_short_cooldown_fraction < 1.0):
                raise ValueError(
                    "grad reuse short-horizon fractions must satisfy "
                    "0 < warmup_fraction < cooldown_fraction < 1"
                )

        if self.period < 2:
            raise ValueError("grad reuse period must be >= 2 (1 reuses nothing)")
        if self.period > 2 and not self.allow_long_period:
            raise ValueError(
                "grad reuse period %d asks for a gradient %d steps stale, and "
                "the staleness probe measured cos 0.5141 at two steps and "
                "-0.2658 at four - the reused gradient would point AWAY from "
                "the descent direction. Set allow_long_period to override, and "
                "re-measure before believing the result."
                % (self.period, self.period - 1))
        if self.batched_check and self.period != 2:
            raise ValueError(
                "grad reuse batched_check requires period=2 so two reuse "
                "iterations can never be consecutive")
        if self.warmup < 0:
            raise ValueError("grad reuse warmup must be >= 0")
        if self.cooldown < 0:
            raise ValueError("grad reuse cooldown must be >= 0")
        if 0 < self.cooldown <= self.warmup:
            # Inert rather than dangerous - nothing would ever reuse -
            # but it always means the two were read the wrong way round.
            raise ValueError(
                "grad reuse cooldown (%d) at or below warmup (%d) leaves "
                "no iterations to reuse: warmup protects the OPENING of a "
                "frame and cooldown the CLOSE, so cooldown must be the "
                "larger." % (self.cooldown, self.warmup))

        self.frames = 0
        self.rendered = 0
        self.reused = 0
        # Snapshot of (reused, rendered) at the last reset_frame() call, so
        # the PER-FRAME counts printed there are a delta against the running
        # totals above, not a second set of counters to keep in sync.
        self._frame_reused0 = 0
        self._frame_rendered0 = 0
        self._have_stash = False

        # Adaptive-only state, all reset per frame in reset_frame().
        self._last_fresh_grad = None   # device tensor, the previous FRESH g
        self._pending_cos = None       # device scalar, MIN cos since last check
        self._pending_check = None     # (had_mag, iter_idx, frame_token)
        # Async reuse is forbidden until a current, non-backlogged check has
        # completed. This is deliberately stricter than the sync default.
        self._async_trust_ready = not self.async_check
        # One persistent destination/event avoids a pinned allocation and a
        # CUDA-event construction for every trust check.
        self._async_host = None
        self._async_event = None
        self._frame_token = 0          # rejects late cross-frame decisions
        # Magnitude tracks a [min, max] PAIR, not a single running value:
        # "worse" means further from 1.0 in EITHER direction, so a window
        # that dips to 0.5 on one pair and spikes to 2.0 on another must fail
        # even though neither alone would look extreme against the other.
        self._pending_mag_min = None
        self._pending_mag_max = None
        self._since_check = 0          # fresh-gradient calls since last host read
        self._boundary = None          # current adaptive reuse-eligible boundary
        self._locally_trusted = True   # local_trust mode: live flag, no ratchet
        self._reject_streak = 0        # consecutive rejected trust checks
        self._reject_latched = False   # fresh-only for the rest of this frame
        self._reject_backoff_until = 0 # next real iteration allowed to probe
        self._trust_lease_remaining = 0  # future alternating reuse slots
        self.batched_signals = 0         # GPU readings staged into drains
        self.batched_lease_reuses = 0    # credits actually consumed
        self.reject_latches = 0        # run-level count of latched frames
        self.reject_backoff_skips = 0  # run-level fresh probes suppressed
        # Run-level counters, so a bad reading anywhere is visible in the
        # summary rather than silently averaged away.
        self.adaptive_checks = 0
        self.adaptive_extends = 0
        self.adaptive_freezes = 0
        # Of the freezes above, how many were caused by the magnitude gate
        # SPECIFICALLY (cosine alone would have passed) - the only way to
        # see whether trust_mag_enabled is doing anything at all, since a
        # freeze count alone cannot distinguish the two gates.
        self.adaptive_freezes_mag = 0
        self._boundary_history = []    # end-of-frame boundary, for a spread stat
        self._calibration_complete = self.calibration_frames == 0
        self._calibration_usable = True
        self._calibration_observed = 0
        self._calibration_stops = []
        self._calibration_reasons = {}
        self._calibration_median = None
        self._calibration_anchor = None
        self._calibration_raw_cutoff = None
        self._calibration_short_anchor = None
        self._calibration_short_selected = False
        # A short frame cannot supply enough evidence for the long-frame
        # adaptive gate.  This flag makes should_reuse() apply the calibrated
        # fixed ceiling even when adaptive=True was requested, and the summary
        # says so explicitly rather than reporting an adaptive arm that never
        # had a chance to adapt.
        self._fixed_calibrated_window = False

        if self.enabled:
            print(f"[GradReuse] constructed (period={self.period}, "
                  f"warmup={self.warmup}, cooldown={self.cooldown or 0}, "
                  f"step_scale={self.step_scale:g}, "
                  f"no_accum={self.no_accum}, freeze_m={self.freeze_m}, "
                  f"adaptive={self.adaptive}"
                  + (f", trust_cos={self.trust_cos:g}, "
                     f"check_every={self.check_every} fresh grads"
                     if self.adaptive else "")
                  + (f", trust_mag=[{self.trust_mag_low:g},"
                     f"{self.trust_mag_high:g}]"
                     if self.adaptive and self.trust_mag_enabled else "")
                  + (", local_trust=True (no boundary ratchet)"
                     if self.adaptive and self.local_trust else "")
                  + (f", reject_patience={self.reject_patience}"
                     if self.adaptive and self.reject_patience else "")
                  + (f", reject_backoff={self.reject_backoff} real iters"
                     if self.adaptive and self.reject_backoff else "")
                  + (", async_check=True (fail-closed hidden host sync)"
                     if self.adaptive and self.async_check else "")
                  + (", deferred_check=True (exact next-decision readback)"
                     if self.adaptive and self.deferred_check else "")
                  + (f", batched_check=True (piggybacked drain, "
                     f"lease={self.trust_lease}, period-2 alternating)"
                     if self.adaptive and self.batched_check else "")
                  + (f", calibration_frames={self.calibration_frames}, "
                     f"margin={self.calibration_margin}, "
                     f"round={self.calibration_round}, "
                     f"stat={self.calibration_stat}, "
                     f"exclude={sorted(self.calibration_exclude_reasons)}"
                     + (f", short_horizon=1, phase="
                        f"{self.calibration_phase_boundary}, "
                        f"min_horizon={self.calibration_min_horizon}"
                        if self.calibration_short_horizon else "")
                     if self.calibration_frames else "")
                  + (f", hold_frames={self.hold_frames}"
                     if self.hold_frames else "")
                  + ")", flush=True)
            if (self.adaptive and not self.cooldown
                    and not self.calibration_frames):
                # MEASURED, NOT SPECULATED: GSLAM/fr1_desk, freeze_m=True,
                # trust_cos=0.95 - the strict, "safe" tier by
                # grad_staleness.py's own usable_cos default - still cost
                # -1.0 dB PSNR, -0.013 SSIM and pushed 78% of frames to the
                # full budget (boundary p50/p90 saturating near frame length)
                # relative to the SAME config with cooldown=70 as a ceiling,
                # which matched baseline on every metric. Direction staying
                # correlated is not sufficient evidence that continuing to
                # reuse is safe - it is necessary but not sufficient. The
                # ceiling is not a training wheel to remove once trust_cos is
                # tuned; it is load-bearing on its own. Set cooldown as an
                # absolute ceiling even in adaptive mode, unless this run IS
                # deliberately probing how loose the mechanism can go.
                print("[GradReuse] WARNING: adaptive=True with no cooldown "
                     "ceiling. Measured unsafe even at trust_cos=0.95: real "
                     "PSNR/SSIM loss and a stopping rule that stops firing "
                     "for most frames. Set cooldown to cap the boundary.",
                     flush=True)

    @property
    def calibrating(self):
        return self.enabled and not self._calibration_complete

    @property
    def holding(self):
        """True while an independent frame-prefix reuse hold is active."""
        return self.enabled and self.frames <= self.hold_frames

    def reset_frame(self):
        # PER-FRAME REUSE COUNT, as a delta against the running totals -
        # self.reused/self.rendered only ever accumulate, so the frame that
        # just ended is whatever changed since the last reset_frame() call.
        # Skipped on the very first call (self.frames == 0): there is no
        # previous frame yet to report.
        if self.enabled and self.frames > 0:
            _fr_reused = self.reused - self._frame_reused0
            _fr_rendered = self.rendered - self._frame_rendered0
            _fr_total = _fr_reused + _fr_rendered
            if _fr_total > 0:
                print(f"[GradReuse] frame {self.frames}: {_fr_reused}/{_fr_total} "
                      f"reused ({100.0 * _fr_reused / _fr_total:.1f}%)", flush=True)
        self._frame_reused0 = self.reused
        self._frame_rendered0 = self.rendered
        # The stash NEVER crosses a frame boundary. The map is frozen within a
        # frame but the pose jumps to a new initialisation between frames, so a
        # carried-over gradient would be evaluated at a pose the optimiser is
        # nowhere near.
        self._have_stash = False
        if (self._calibration_complete and self._calibration_usable
                and self.adaptive and not self._fixed_calibrated_window
                and not self.local_trust
                and self._boundary is not None):
            self._boundary_history.append(self._boundary)
        # The trust state is EXACTLY as frame-local as the stash, for the
        # same reason: a gradient from the previous frame says nothing about
        # this one's pose.
        # Keep an in-flight copy alive because its pinned destination cannot
        # be reused safely yet. Advancing the token makes the poll discard the
        # old-frame decision as soon as it lands.
        self._frame_token += 1
        self._last_fresh_grad = None
        self._pending_cos = None
        self._pending_mag_min = None
        self._pending_mag_max = None
        self._since_check = 0
        self._async_trust_ready = not self.async_check
        self._reject_streak = 0
        self._reject_latched = False
        self._reject_backoff_until = 0
        self._trust_lease_remaining = 0
        self._boundary = (self.warmup + self.check_every * self.period
                          if (self.adaptive and self._calibration_complete
                              and self._calibration_usable
                              and not self._fixed_calibrated_window
                              and not self.local_trust)
                          else None)
        # Default TRUSTED until the first check completes, matching the
        # boundary mode's own default (it permits reuse for the first
        # check_every*period iterations before any check has run at all).
        self._locally_trusted = True
        self.frames += 1

    def should_reuse(self, iter_idx, is_last):
        """True when this iteration should skip the render entirely."""
        if not self.enabled or not self._have_stash:
            return False
        if self.holding:
            return False
        # Unlike async_check, this may wait - but only at the exact next
        # decision that the original synchronous .item() would have governed.
        if self.deferred_check:
            self._poll_pending_check(block=True)
        # OPPORTUNISTIC, NEVER BLOCKING, AND FAIL-CLOSED: consume a landed
        # result, but render fresh while the copy is still pending or when
        # newer evidence accumulated behind the completed check. The latter
        # must itself be checked before the older result may authorize reuse.
        if self.async_check:
            self._poll_pending_check()
        if self._reject_latched:
            return False
        if self.async_check:
            if self._pending_check is not None or not self._async_trust_ready:
                return False
        if self.calibrating:
            return False
        if self.calibration_frames and not self._calibration_usable:
            return False
        if iter_idx < self.warmup:
            return False
        if self._fixed_calibrated_window:
            if self.cooldown and iter_idx >= self.cooldown:
                return False
        elif self.adaptive:
            if self.batched_check:
                # A completed signal-buffer drain is the only authority that
                # can grant credits. Starting a frame, a failed check, or an
                # expired lease therefore fails closed to fresh rendering.
                if self._trust_lease_remaining <= 0:
                    return False
            elif self.local_trust:
                # NO RATCHET: consults only the most recent check's outcome.
                # A failure stops reuse THIS iteration, not just future
                # extension - and a later success turns it back on. See
                # local_trust's doc in __init__ for why this differs from
                # the boundary mode below.
                if not self._locally_trusted:
                    return False
            # _boundary MOVES: note_fresh_grad() pushes it out on a
            # trusted check and leaves it alone on a failed one. cooldown,
            # if set, still applies as an absolute ceiling (see
            # note_fresh_grad).
            elif self._boundary is not None and iter_idx >= self._boundary:
                return False
            if self.cooldown and iter_idx >= self.cooldown:
                # Still an absolute ceiling in local_trust mode too - a
                # persistently trusted late-frame check should not be able
                # to bypass it.
                return False
        elif self.cooldown and iter_idx >= self.cooldown:
            # The polish phase: render every iteration so the step can
            # decay and the stopping rule can see it.
            return False
        # NEVER on the iteration whose loss the frame may commit from. SplaTAM
        # keeps the best-loss pose, and a reuse iteration produces no loss to
        # compare - so committing from one would commit a pose no render ever
        # scored. That is the same class of error as sparse's final_dense bug,
        # which cost 7.96 cm against 4.04.
        if is_last:
            return False
        return (iter_idx % self.period) != 0

    def note_rendered(self):
        self.rendered += 1
        self._have_stash = True

    def note_fresh_grad(self, g, iter_idx):
        """Feed a FRESH tangent gradient into the trust signal. Adaptive only.

        Call this AFTER a render iteration's gradient is computed - never on
        a reuse iteration, whose gradient is a copy and would make the trust
        signal compare a vector against itself. Cheap and capture-safe by
        construction: cos is computed on-device every call, and the only host
        read happens once every check_every calls, so it costs nothing on
        the calls in between. In batched_check mode there is no read here:
        the method returns a one-scalar cosine signal (three scalars when the
        magnitude gate is enabled) for ESSignalBuffer to attach to this
        iteration's already-scheduled drain.

        g may be None (no preconditioner, or grad missing this iteration) -
        treated as no evidence either way, not as distrust.
        """
        if (not self.adaptive or self._fixed_calibrated_window or self.calibrating
                or self.holding
                or not self._calibration_usable or g is None
                or (self.cooldown and iter_idx >= self.cooldown)
                or self._reject_latched):
            return
        if (self.reject_backoff
                and iter_idx < self._reject_backoff_until):
            # Reuse is already disabled by _locally_trusted=False. Keep the
            # freshest possible comparison baseline, but deliberately do no
            # dot/norm work and no host read until the recovery deadline.
            # Clearing the window is essential: the eventual recovery probe
            # must compare adjacent fresh gradients, not retain the rejected
            # pair that started the backoff.
            self._last_fresh_grad = g.detach().clone()
            self._pending_cos = None
            self._pending_mag_min = None
            self._pending_mag_max = None
            self._since_check = 0
            self.reject_backoff_skips += 1
            return
        if self._last_fresh_grad is not None:
            # ONE DOT PRODUCT AND TWO NORMS, no .item(). This is the whole
            # per-iteration cost; the sync is paid only at the check below.
            #
            # ACCUMULATED AS A RUNNING MIN, not overwritten. A single call
            # between checks would only ever see the LATEST pair, and a
            # transient disagreement can be masked by the pair either side
            # of it happening to agree with EACH OTHER (both post-flip, say)
            # - a real failure mode found by utils/test_grad_reuse_adaptive.py
            # simulating an it20-shaped discontinuity. Taking the min over
            # the whole batch window means any bad pair inside it is seen,
            # still with no extra sync - torch.minimum is on-device.
            a, b = g, self._last_fresh_grad
            an, bn = a.norm(), b.norm()
            cos_now = torch.dot(a, b) / (an * bn + 1e-12)
            self._pending_cos = (cos_now if self._pending_cos is None
                                 else torch.minimum(self._pending_cos, cos_now))
            if self.trust_mag_enabled:
                # SAME PAIR, an/bn ALREADY COMPUTED ABOVE - no extra norm.
                # Tracked as a running [min, max], not a running min alone
                # like cos: "worse" means further from 1.0 in EITHER
                # direction, so a window that dips to 0.5 on one pair and
                # spikes to 2.0 on another must fail even though neither
                # alone looks extreme against the other. Still on-device,
                # still no extra sync until the check below.
                mag_now = an / (bn + 1e-12)
                self._pending_mag_min = (
                    mag_now if self._pending_mag_min is None
                    else torch.minimum(self._pending_mag_min, mag_now))
                self._pending_mag_max = (
                    mag_now if self._pending_mag_max is None
                    else torch.maximum(self._pending_mag_max, mag_now))
        self._last_fresh_grad = g.detach().clone()
        self._since_check += 1
        if self._since_check < self.check_every or self._pending_cos is None:
            return
        if self.batched_check:
            self._since_check = 0
            if self.trust_mag_enabled and self._pending_mag_min is not None:
                signal = torch.stack([
                    self._pending_cos,
                    self._pending_mag_min,
                    self._pending_mag_max,
                ])
            else:
                # A view, not a newly-filled validity tensor. Default buffer
                # rows are NaN, so a finite first column is its own validity
                # marker and the common cosine-only path adds no allocation.
                signal = self._pending_cos.reshape(1)
            self._pending_cos = None
            self._pending_mag_min = None
            self._pending_mag_max = None
            self.batched_signals += 1
            return signal
        # ASYNC PATH: hand the window off instead of blocking on it.
        #
        # Poll first - the common case is the PREVIOUS check already landed
        # (copies this small are typically far faster than a few more
        # iterations of Python-level loop overhead), clearing the slot
        # before this window needs it.
        #
        # If a check is STILL in flight despite that, do NOT overwrite or
        # drop it and do NOT reset _since_check/_pending_cos - just return
        # and keep accumulating. This widens the window by however many
        # more fresh grads it takes for the backlog to clear, rather than
        # ever running two async copies at once or silently discarding
        # evidence. Degrades toward the synchronous behaviour's own
        # worst case (a late check) under backlog, never toward a race.
        if self.async_check or self.deferred_check:
            self._poll_pending_check(block=self.deferred_check)
            if self._pending_check is not None:
                return
            # A future check policy may clear the accumulator while applying
            # the completed result. Guard the packaging step explicitly: this
            # is the state transition behind the old None.reshape crash.
            if self._pending_cos is None:
                return
            self._since_check = 0
            self.adaptive_checks += 1
            if self.trust_mag_enabled and self._pending_mag_min is not None:
                stacked = torch.stack(
                    [self._pending_cos, self._pending_mag_min,
                     self._pending_mag_max])
                had_mag = True
            else:
                stacked = self._pending_cos.reshape(1)
                had_mag = False
            if (self._async_host is None
                    or self._async_host.shape != stacked.shape
                    or self._async_host.dtype != stacked.dtype):
                self._async_host = torch.empty(
                    stacked.shape, dtype=stacked.dtype, device="cpu",
                    pin_memory=True)
            if self._async_event is None:
                self._async_event = torch.cuda.Event()
            self._async_trust_ready = False
            self._async_host.copy_(stacked, non_blocking=True)
            self._async_event.record()
            self._pending_check = (had_mag, iter_idx, self._frame_token)
            self._pending_cos = None
            self._pending_mag_min = None
            self._pending_mag_max = None
            return
        # SYNCHRONOUS PATH (default): unchanged from before async_check
        # existed - blocks here, applies immediately.
        self._since_check = 0
        self.adaptive_checks += 1
        # THE ONE HOST SYNC (still one, not three): cos and the magnitude
        # min/max, when tracked, are read together in a single stacked
        # .tolist() rather than three separate .item() calls, each of which
        # would pay its own device-to-host round trip.
        if self.trust_mag_enabled and self._pending_mag_min is not None:
            cos, mag_min, mag_max = torch.stack(
                [self._pending_cos, self._pending_mag_min,
                 self._pending_mag_max]).tolist()
        else:
            cos = float(self._pending_cos.item())
            mag_min = mag_max = None
        self._pending_cos = None  # fresh accumulators for the next window
        self._pending_mag_min = None
        self._pending_mag_max = None
        self._apply_check(cos, mag_min, mag_max, iter_idx)

    def note_batched_check(self, cos, mag_min, mag_max, iter_idx):
        """Consume one exact trust result from an existing host drain.

        SplaTAM calls this only after ESSignalBuffer has copied a complete
        batch to the host. The newest valid signal in that batch predicts the
        next block; no pending result and no speculative reuse are involved.
        """
        if not self.batched_check:
            raise RuntimeError("note_batched_check requires batched_check")
        if self._reject_latched:
            return
        self.adaptive_checks += 1
        self._apply_check(cos, mag_min, mag_max, int(iter_idx))

    def reset_trust_lease(self):
        """Fail closed across an objective/optimizer/anomaly boundary."""
        if self.batched_check:
            self._trust_lease_remaining = 0
            self._locally_trusted = False
            self._last_fresh_grad = None
            self._pending_cos = None
            self._pending_mag_min = None
            self._pending_mag_max = None
            self._since_check = 0

    def _poll_pending_check(self, block=False):
        """Consume a copied check result. Async polling never blocks;
        deferred-exact consumption waits only at the next reuse decision."""
        if self._pending_check is None:
            return False
        if block:
            self._async_event.synchronize()
        elif not self._async_event.query():
            return False
        had_mag, iter_idx, frame_token = self._pending_check
        if had_mag:
            cos, mag_min, mag_max = self._async_host.tolist()
        else:
            cos = self._async_host[0].item()  # CPU tensor - no device sync
            mag_min = mag_max = None
        self._pending_check = None
        if frame_token != self._frame_token:
            self._async_trust_ready = False
            return True
        self._apply_check(cos, mag_min, mag_max, iter_idx)
        # If fresh gradients arrived while this copy was in flight, its
        # answer is already stale. Keep rendering until that accumulated
        # evidence has completed its own check. With no backlog, this result
        # is current and may safely authorize the next reuse candidate.
        self._async_trust_ready = self._pending_cos is None
        return True

    def _apply_check(self, cos, mag_min, mag_max, iter_idx):
        """Install one check's trust decision - shared by the synchronous
        and async paths so they can never diverge in how a (cos, mag_min,
        mag_max) reading turns into a boundary/local_trust update."""
        cos_trusted = cos == cos and cos >= self.trust_cos  # cos==cos rejects NaN
        mag_trusted = (not self.trust_mag_enabled) or (
            mag_min is not None and mag_min == mag_min and mag_max == mag_max
            and mag_min >= self.trust_mag_low and mag_max <= self.trust_mag_high)
        trusted = cos_trusted and mag_trusted
        if trusted:
            self._reject_streak = 0
            self._reject_backoff_until = 0
        else:
            self._reject_streak += 1
            if self.reject_backoff:
                self._reject_backoff_until = iter_idx + self.reject_backoff
        if self.batched_check:
            # Replace, never add: one unusually stable block cannot build an
            # unbounded reservoir of stale authority. period=2 remains the
            # final scheduler, so lease=2 means F,R,F,R, never F,R,R.
            self._locally_trusted = trusted
            self._trust_lease_remaining = (
                self.trust_lease if trusted else 0)
            if trusted:
                self.adaptive_extends += 1
            else:
                self.adaptive_freezes += 1
                if self.trust_mag_enabled and cos_trusted and not mag_trusted:
                    self.adaptive_freezes_mag += 1
        elif self.local_trust:
            # LIVE FLAG, NO RATCHET: set directly from this window's result.
            # A failure turns reuse off immediately (should_reuse() checks
            # this flag every iteration, not a frozen boundary position),
            # and a later success turns it back on.
            self._locally_trusted = trusted
            if trusted:
                self.adaptive_extends += 1
            else:
                self.adaptive_freezes += 1
                if self.trust_mag_enabled and cos_trusted and not mag_trusted:
                    self.adaptive_freezes_mag += 1
        elif trusted:
            self.adaptive_extends += 1
            target = iter_idx + self.check_every * self.period
            self._boundary = (target if self._boundary is None
                              else max(self._boundary, target))
            if self.cooldown:
                self._boundary = min(self._boundary, self.cooldown)
        else:
            # FREEZE, DO NOT SHRINK. The boundary already extended past this
            # point stays extended - iterations already scheduled to reuse
            # keep doing so - but no FURTHER iterations are added. A bad
            # reading (the it20 DIAG-switch artefact, for instance) just ends
            # reuse a little early for this one frame: the safe direction to
            # fail in, not a dangerous one.
            self.adaptive_freezes += 1
            if self.trust_mag_enabled and cos_trusted and not mag_trusted:
                self.adaptive_freezes_mag += 1
        if (not trusted and self.reject_patience
                and self._reject_streak >= self.reject_patience
                and not self._reject_latched):
            self._reject_latched = True
            self.reject_latches += 1
            self._locally_trusted = False
            self._trust_lease_remaining = 0
            # No more trust work is useful in this frame. Drop any newer
            # accumulated evidence; _pending_check is already clear because
            # _apply_check is called only after its value has been consumed.
            self._pending_cos = None
            self._pending_mag_min = None
            self._pending_mag_max = None
            self._since_check = 0

    def note_frame_stop(self, iterations, reason="unknown"):
        """Observe a run-in frame and select a conservative reuse ceiling.

        The standard rule is
        ``floor((anchor_stop - margin) / round) * round``. A minimum of one
        rounding interval beyond warmup keeps the calibrated reuse window
        non-empty if the observed stopping point is unusually early.

        With calibration_short_horizon enabled, an earliest eligible stop
        that leaves less than calibration_min_horizon after the configured
        phase boundary selects a fixed, period-aligned fractional window
        instead.  See the module docstring; this is the SplaTAM 25 -> w6/c20
        path and is opt-in so existing long-frame callers do not move.
        """
        if not self.calibrating:
            return
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("grad reuse calibration stop must be >= 1 iteration")
        reason = str(reason)
        self._calibration_observed += 1
        self._calibration_reasons[reason] = self._calibration_reasons.get(reason, 0) + 1
        if reason not in self.calibration_exclude_reasons:
            self._calibration_stops.append(iterations)
        if self._calibration_observed < self.calibration_frames:
            return

        self._calibration_complete = True
        if not self._calibration_stops:
            self._calibration_usable = False
            reasons = ",".join(
                f"{key}={value}"
                for key, value in sorted(self._calibration_reasons.items())
            )
            print(
                "[GradReuse] calibration complete with no eligible stopping "
                f"frames over {self.calibration_frames} observations; "
                f"reasons={reasons}; gradient reuse remains disabled",
                flush=True,
            )
            return

        median_stop = float(np.median(self._calibration_stops))
        anchor_stop = (
            float(min(self._calibration_stops))
            if self.calibration_stat == "min" else median_stop
        )
        earliest_stop = float(min(self._calibration_stops))
        self._calibration_median = median_stop
        self._calibration_anchor = anchor_stop
        self._calibration_short_anchor = earliest_stop

        if (self.calibration_short_horizon
                and earliest_stop - self.calibration_phase_boundary
                < self.calibration_min_horizon):
            _p = self.period
            short_warmup = int(np.floor(
                earliest_stop * self.calibration_short_warmup_fraction / _p
            ) * _p)
            short_warmup = max(
                self.calibration_short_min_warmup_periods * _p,
                short_warmup,
            )
            short_cooldown = int(np.floor(
                earliest_stop * self.calibration_short_cooldown_fraction / _p
            ) * _p)
            if (short_cooldown < short_warmup + _p
                    or short_cooldown >= earliest_stop):
                # There is no complete period-aligned reuse slot plus a fresh
                # tail inside the observed stop.  Silently clamping above the
                # stop would advertise an enabled arm that can never fire, so
                # disable it and make the reason explicit.
                self._calibration_usable = False
                print(
                    "[GradReuse] short-horizon calibration found no usable "
                    f"window before earliest stop {earliest_stop:g}; "
                    "gradient reuse remains disabled",
                    flush=True,
                )
                return
            self.warmup = short_warmup
            self.cooldown = short_cooldown
            self._calibration_raw_cutoff = short_cooldown
            self._calibration_short_selected = True
            self._fixed_calibrated_window = True
            print(
                f"[GradReuse] selected_warmup={self.warmup} "
                f"selected_cooldown={self.cooldown} mode=short",
                flush=True,
            )
            return

        raw_cutoff = int(
            np.floor((anchor_stop - self.calibration_margin)
                     / self.calibration_round) * self.calibration_round
        )
        self.cooldown = max(
            self.warmup + self.calibration_round,
            raw_cutoff,
        )
        self._calibration_raw_cutoff = raw_cutoff
        print(
            f"[GradReuse] selected_cooldown={self.cooldown}",
            flush=True,
        )

    def note_reuse_candidate(self):
        """Consume the authority for one eligible reuse decision.

        The speedup-breakdown control renders the candidate iteration while
        keeping the trust checker live.  It calls this method without calling
        ``note_reused`` so the lease cadence matches the applying arm while
        the accounting still truthfully records a rendered iteration.
        """
        if self.batched_check:
            if self._trust_lease_remaining <= 0:
                raise RuntimeError(
                    "batched gradient reuse consumed an empty trust lease")
            self._trust_lease_remaining -= 1
            self.batched_lease_reuses += 1

    def note_reused(self):
        self.reused += 1
        self.note_reuse_candidate()

    def summary(self):
        if not self.enabled:
            return "Gradient reuse: disabled"
        total = self.rendered + self.reused
        if not total:
            if self.calibration_frames:
                return (
                    "Gradient reuse: calibration armed; reuse disabled for "
                    f"the first {self.calibration_frames} tracked frames, "
                    f"warmup={self.warmup}, margin={self.calibration_margin}, "
                    f"round={self.calibration_round}"
                )
            return "Gradient reuse: enabled but never fired"
        lines = [
            f"Gradient reuse: period={self.period}, warmup={self.warmup}, "
            f"cooldown={self.cooldown or 0}; "
            f"{self.reused}/{total} iterations reused "
            f"({100.0 * self.reused / total:.1f}%), "
            f"{self.rendered} rendered over {self.frames} frames, "
            f"step_scale={self.step_scale:g}, "
            f"no_accum={self.no_accum}, freeze_m={self.freeze_m}, "
            f"adaptive={self.adaptive} "
            f"-- READ TRACKING s/FRAME, NOT ms/iter: half the iterations do no "
            f"render, so ms/iter improves by construction and is not the result"
        ]
        if self.hold_frames:
            lines.append(
                f"  bootstrap hold: reuse disabled for the first "
                f"{self.hold_frames} tracked frames; no reuse parameters "
                "were fitted or changed")
        if self.adaptive and not self._fixed_calibrated_window:
            _checks = self.adaptive_checks
            _ext_pct = (100.0 * self.adaptive_extends / _checks
                       if _checks else float("nan"))
            _hist = sorted(self._boundary_history)
            _p = (lambda q: _hist[min(len(_hist) - 1, int(q * len(_hist)))]
                 if _hist else float("nan"))
            _readback = ("batched-lease%d" % self.trust_lease
                         if self.batched_check else
                         "async-fail-closed" if self.async_check else
                         "deferred-exact" if self.deferred_check else "sync")
            _cadence = (
                f"check_every={self.check_every} fresh grads/signal, "
                "host decisions at existing drains"
                if self.batched_check else
                f"check_every={self.check_every} fresh grads "
                f"({self.check_every * self.period} real iters/check)")
            lines.append(
                f"  adaptive: trust_cos={self.trust_cos:g}, "
                f"{_cadence}, "
                f"readback={_readback}; "
                f"checks={_checks}, extended={self.adaptive_extends} "
                f"({_ext_pct:.1f}%), frozen={self.adaptive_freezes}"
                + (f" (of which {self.adaptive_freezes_mag} were "
                   f"cos-trusted but magnitude-rejected)"
                   if self.trust_mag_enabled else "")
                + (f"; reject circuit={self.reject_patience} consecutive, "
                   f"latched={self.reject_latches} frames"
                   if self.reject_patience else "")
                + (f"; reject backoff={self.reject_backoff} real iters, "
                   f"suppressed={self.reject_backoff_skips} probes"
                   if self.reject_backoff else "")
                + (f"; end-of-frame boundary p10/p50/p90="
                   f"{_p(0.1):.0f}/{_p(0.5):.0f}/{_p(0.9):.0f}"
                   if not self.local_trust else
                   " (batched lease: no boundary to report)"
                   if self.batched_check else
                   " (local_trust: no boundary to report - reuse tracks "
                   "the live flag every check instead)"))
            if self.batched_check:
                lines.append(
                    f"    batched trust: {self.batched_signals} GPU signals "
                    f"piggybacked, lease={self.trust_lease}, "
                    f"credits consumed={self.batched_lease_reuses}; "
                    "period=2 forbids consecutive reuse")
            if self.trust_mag_enabled:
                lines.append(
                    f"    trust_mag=[{self.trust_mag_low:g},"
                    f"{self.trust_mag_high:g}] gates the SAME checks above, "
                    f"not a separate schedule")
        if self.calibration_frames:
            observed = self._calibration_observed
            eligible = len(self._calibration_stops)
            if self._calibration_complete:
                reasons = ",".join(
                    f"{key}={value}"
                    for key, value in sorted(self._calibration_reasons.items())
                )
                if self._calibration_usable:
                    vals = np.asarray(
                        self._calibration_stops, dtype=np.float64
                    )
                    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
                    if self._calibration_short_selected:
                        lines.append(
                            f"  calibration: {observed}/"
                            f"{self.calibration_frames} frames, {eligible} "
                            f"eligible stops, stop p10/p50/p90="
                            f"{p10:.0f}/{p50:.0f}/{p90:.0f}; short horizon "
                            f"(earliest {self._calibration_short_anchor:.0f} "
                            f"- phase {self.calibration_phase_boundary} < "
                            f"{self.calibration_min_horizon}), selected fixed "
                            f"warmup/cooldown={self.warmup}/{self.cooldown}; "
                            f"reasons {reasons}"
                        )
                    else:
                        lines.append(
                            f"  calibration: {observed}/"
                            f"{self.calibration_frames} frames, {eligible} "
                            f"eligible stops, stop p10/p50/p90="
                            f"{p10:.0f}/{p50:.0f}/{p90:.0f}; "
                            f"floor(({self.calibration_stat}-"
                            f"{self.calibration_margin})/"
                            f"{self.calibration_round})*"
                            f"{self.calibration_round}="
                            f"{self._calibration_raw_cutoff}, selected "
                            f"cooldown={self.cooldown}; reasons {reasons}"
                        )
                else:
                    lines.append(
                        f"  calibration: {observed}/{self.calibration_frames} "
                        "frames, 0 eligible stops; reuse disabled; reasons "
                        f"{reasons}"
                    )
            else:
                lines.append(
                    f"  calibration: {observed}/{self.calibration_frames} "
                    f"frames, {eligible} eligible stops "
                    "(reuse still disabled)"
                )
        return "\n".join(lines)


def stash_grads(tensors):
    """Copy the pose gradients before the step zeroes them.

    The preconditioner's step clears .grad with .zero_() so the addresses stay
    stable for a CUDA graph replay. So the gradient must be copied BETWEEN
    backward() and the step, not after it - after it, there is nothing left to
    reuse.
    """
    return [None if t.grad is None else t.grad.detach().clone()
            for t in tensors]


def restore_grads(tensors, stash):
    """Write a stashed gradient back so the step path finds it as usual.

    copy_ into the existing tensor rather than rebinding .grad: the same
    address-stability requirement the zeroing above is written for.
    """
    if stash is None:
        return False
    for t, s in zip(tensors, stash):
        if s is None:
            continue
        if t.grad is None:
            t.grad = s.clone()
        else:
            t.grad.copy_(s)
    return True


def config_from_env(base=None):
    """GRAD_REUSE=<period> turns it on; 0 or unset leaves it off."""
    cfg = dict(base or {})
    raw = os.environ.get("GRAD_REUSE")
    if raw is None:
        return cfg
    period = int(raw)
    if period <= 1:
        cfg["enabled"] = False
        return cfg
    cfg["enabled"] = True
    cfg["period"] = period
    _hold = os.environ.get("GRAD_REUSE_HOLD_FRAMES")
    if _hold is not None:
        cfg["hold_frames"] = int(_hold)
    # GRAD_REUSE_WARMUP: how many iterations at the START of each frame
    # always render.
    #
    # Worth sweeping because the pose does not move at a constant rate
    # through a frame. The step profile puts it0-4 at 0.41x the reference
    # and it10-19 at 0.20x - roughly twice the motion early - and a stale
    # gradient is worth least exactly where the pose is moving most. The
    # staleness probe measured cos 0.9473 AVERAGED over its windows, so a
    # higher early figure and a lower late one are both hiding inside it.
    #
    # Each extra warmup iteration costs a render, so this trades speedup
    # for accuracy directly - read both columns, not one.
    _w = os.environ.get("GRAD_REUSE_WARMUP")
    if _w is not None:
        cfg["warmup"] = int(_w)
    _c = os.environ.get("GRAD_REUSE_COOLDOWN")
    if _c is not None:
        cfg["cooldown"] = int(_c)
    _ss = os.environ.get("GRAD_REUSE_STEP_SCALE")
    if _ss is not None:
        cfg["step_scale"] = float(_ss)
    if os.environ.get("GRAD_REUSE_NOACC", "0") not in ("0", "", "false", "False"):
        cfg["no_accum"] = True
    if os.environ.get("GRAD_REUSE_FREEZE_M", "0") not in ("0", "", "false", "False"):
        cfg["freeze_m"] = True
    # GRAD_REUSE_ADAPTIVE=1 replaces the fixed cooldown with a per-frame
    # boundary driven by the trust signal - see the module docstring. Setting
    # GRAD_REUSE_COOLDOWN alongside it keeps cooldown as a ceiling the
    # adaptive boundary cannot cross, rather than switching it off.
    if os.environ.get("GRAD_REUSE_ADAPTIVE", "0") not in ("0", "", "false", "False"):
        cfg["adaptive"] = True
    # GRAD_REUSE_LOCAL_TRUST=1 drops the boundary ratchet - see
    # GradReuse.__init__'s local_trust docstring. Off by default: nothing
    # changes unless asked.
    if os.environ.get("GRAD_REUSE_LOCAL_TRUST", "0") not in ("0", "", "false", "False"):
        cfg["local_trust"] = True
    # GRAD_REUSE_ASYNC_CHECK=1 hides the per-check host sync instead of
    # paying it inline - see GradReuse.__init__'s async_check docstring. Off
    # by default: nothing changes unless asked.
    if os.environ.get("GRAD_REUSE_ASYNC_CHECK", "0") not in ("0", "", "false", "False"):
        cfg["async_check"] = True
    # GRAD_REUSE_DEFER_CHECK=1 starts the non-blocking copy at the check
    # boundary but waits for its exact result at the next reuse decision.
    # Unlike ASYNC_CHECK, it never substitutes a fresh render for a decision.
    if os.environ.get("GRAD_REUSE_DEFER_CHECK", "0") not in (
            "0", "", "false", "False"):
        cfg["deferred_check"] = True
    if os.environ.get("GRAD_REUSE_BATCHED_CHECK", "0") not in (
            "0", "", "false", "False"):
        cfg["batched_check"] = True
    _lease = os.environ.get("GRAD_REUSE_TRUST_LEASE")
    if _lease is not None:
        cfg["trust_lease"] = int(_lease)
    # GRAD_REUSE_REJECT_PATIENCE=N stops both reuse and further trust checks
    # for the rest of a frame after N consecutive rejected checks. It resets
    # at every frame boundary; zero/unset disables it.
    _rp = os.environ.get("GRAD_REUSE_REJECT_PATIENCE")
    if _rp is not None:
        cfg["reject_patience"] = int(_rp)
    # GRAD_REUSE_REJECT_BACKOFF=N keeps rendering fresh after a rejected
    # synchronous local-trust check and waits N real iterations before the
    # next recovery probe. Zero/unset preserves the exact existing cadence.
    _rb = os.environ.get("GRAD_REUSE_REJECT_BACKOFF")
    if _rb is not None:
        cfg["reject_backoff"] = int(_rb)
    _tc = os.environ.get("GRAD_REUSE_TRUST_COS")
    if _tc is not None:
        cfg["trust_cos"] = float(_tc)
    # GRAD_REUSE_TRUST_MAG=1 adds the magnitude-ratio gate alongside cosine -
    # see GradReuse.__init__ and note_fresh_grad for the mechanism. Off by
    # default: nothing changes unless asked.
    if os.environ.get("GRAD_REUSE_TRUST_MAG", "0") not in ("0", "", "false", "False"):
        cfg["trust_mag_enabled"] = True
    _tml = os.environ.get("GRAD_REUSE_TRUST_MAG_LOW")
    if _tml is not None:
        cfg["trust_mag_low"] = float(_tml)
    _tmh = os.environ.get("GRAD_REUSE_TRUST_MAG_HIGH")
    if _tmh is not None:
        cfg["trust_mag_high"] = float(_tmh)
    _ce = os.environ.get("GRAD_REUSE_CHECK_EVERY")
    if _ce is not None:
        cfg["check_every"] = int(_ce)
    return cfg


class RenderClock:
    """Renumber tracking iterations so only RENDERED ones advance the clock.

    WHY. The windowed stopper (utils/windowed_convergence.py) reads its evidence
    on the ITERATION axis: check cadence, patience, the energy window, the phase
    start. With gradient reuse a reused iteration is not new evidence - its loss
    is the previous one and its step comes from a stale gradient - yet it still
    advances that axis. Measured on MonoGS fr1 (cap 50, 200 frames): per
    iteration the reuse arm's step norm and energy ratio lag the control; per
    RENDER they are equal within 1-3% in loss. The stopper was reading dilution
    as slow convergence, which is why saved renders were handed back as extra
    iterations.

    HOW. `note(reused)` records each iteration in order. `map_row(it, step)`
    always returns None for a reused row. In the default, conservative mode it
    banks that tangent step and adds it to the next rendered row. With
    bank_steps=False it deliberately drops the reused row from convergence
    evidence: the optimiser still takes the full stale-gradient step, but only
    a genuinely rendered loss/step pair advances the stopping rule. The latter
    is an accuracy-sensitive experiment, not a claim that the pose did not
    move. `before(it)` is the number of renders completed before iteration
    `it`, which is the correct phase-start coordinate on either render clock.
    """

    def __init__(self, bank_steps=True):
        self.bank_steps = bool(bank_steps)
        self._cum = [0]
        self._pending = None

    def note(self, reused):
        self._cum.append(self._cum[-1] + (0 if reused else 1))

    def before(self, it):
        return self._cum[int(it)]

    def map_row(self, it, step):
        it = int(it)
        step = np.asarray(step, dtype=np.float64)
        if self._cum[it + 1] == self._cum[it]:
            if self.bank_steps:
                self._pending = (step if self._pending is None
                                 else self._pending + step)
            return None
        if self.bank_steps and self._pending is not None:
            step = step + self._pending
            self._pending = None
        return self._cum[it + 1] - 1, step
