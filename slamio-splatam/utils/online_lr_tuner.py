"""Within-run acquisition-lr tuner: no ground truth, one deployed pass.

profiling/lr_probe.py decides REDUCE/KEEP/AMBIGUOUS from two SEPARATE full
runs (arms at k and k/2), paired frame-by-frame. That is not available at
deployment: there is one run, not two, and no ground truth to fall back on
for cap-bound scenes. This reuses the same decision rule - mean delta / SE /
t-stat, sign of the delta - inside ONE run, by INTERLEAVING the two arms on
adjacent frames instead of running them in consecutive blocks.

WHY INTERLEAVED, NOT BLOCKS. The first version ran 8 frames at k, then 8 at
k/2, and compared the block means UNPAIRED. Then "k/2 is faster" and "frames
9-16 are easier than frames 1-8" are the same observation: the arm order IS the
scene phase, and a sequence's opening is not its middle. Now the frames are
grouped into PAIRS (t, t+1), one at the reference k and one at the candidate
k/2, and the test is on the per-pair difference:

  * Scene difficulty that drifts slowly cancels inside a pair - both frames
    see nearly the same motion and texture. This is the same argument that
    made the offline probe paired, but here the partner is the NEXT frame of
    the same run rather than the same frame of a second run, so it costs no
    extra process, no extra startup, and needs no ground truth.
  * WHICH ARM GOES FIRST alternates pair to pair (ref,cand / cand,ref / ...).
    A first-vs-second-frame effect (warm-up, carried optimiser state, a
    constant-velocity pose guess that is better on one side of a pair) adds
    +delta in one order and -delta in the other, and cancels exactly over an
    even number of pairs. So looks are taken at even pair counts only. This
    is the lesson of profiling/ab_counterbalance_order: alternating arms
    between RUNS is not enough, the position within the pair matters.

SEQUENTIAL LOOKS, SMALL-SAMPLE CRITICAL VALUES. With 6-12 pairs the normal
|t|>2 is badly anti-conservative (df=5 needs 2.57 at the 5% level), and
peeking every two pairs inflates it further. The threshold is therefore a
Student-t critical value at a constant nominal 2% two-sided level per look
(Pocock's constant-boundary design: ~5% overall across up to four looks).
Pass t_threshold to override it with a fixed number.

The rule per rung:  |t| >= crit  ->  REDUCE (t<0: accept k/2, halve again) or
KEEP (t>0: freeze at the reference k).  Below crit, keep collecting pairs up to
max_pairs, and if it is STILL below crit that is AMBIGUOUS and freezes at the
larger, safer value - copied from the probe, where the measured asymmetry was
that a wrong REDUCE cost far more (room0's ATE +29% when a real effect was
missed) than a missed REDUCE (fr1's +7% iterations).

FRAME COST. A rung takes 2*min_pairs .. 2*max_pairs frames (12-24 at the
default block_frames=12). There is no separate baseline block any more: the
reference k is measured in the same pairs as the candidate, so the first rung
already compares k=1 with k=0.5 from frame one.

CAP-BOUND RUNGS FALL BACK TO POSE ERROR AGAINST GROUND TRUTH, WHEN THE CALLER
HAS IT. GSLAM/replica_room0 runs ~100% at cap even at 3x its production
budget, so it/frame is a constant there and carries no signal - exactly the
condition profiling/lr_probe.py's offline probe already handles by comparing
cam_trans_err instead. This is not usually available "online" in the sense of
a real deployment with no ground truth, BUT this codebase is a benchmarking
harness: every sequence's true trajectory is loaded for tracking already
(GSLAM's dataset loader hands back gt_c2w per frame), so a run here CAN report
its own pose error, same as the offline probe reads from the log after the
fact. Pass at_cap and pose_err to observe() to enable it; leave them out (as
MonoGS and SplaTAM do) and a cap-bound rung still declines exactly as before,
freezing at the reference k rather than trust a diluted it/frame comparison -
see cap_limit, the same 50% threshold and the same reasoning as lr_probe.py.
The pose-error fallback activates only when EVERY pair in the rung has a
reading; one missing frame falls back to declining rather than average over a
hole.

POSE_ERR: PASS A DELTA, NOT A LEVEL - THIS IS THE PART THAT ACTUALLY MATTERS.
pose_err is measured against a FIXED ground-truth trajectory, so its LEVEL
carries whatever error has accumulated by that point from EVERY prior frame,
under BOTH arms - unlike it/frame, which is frame-local (frame N's iteration
count depends on frame N's own convergence, not frame N-1's). Comparing raw
LEVELS under alternating single-frame interleaving is a DILUTED, ORDER-
CONFOUNDED estimator of the true effect, provably so:

    Two readings of ONE shared, evolving error process, ref-then-cand or
    cand-then-ref depending on which pair. With a per-step effect (cand's
    increment minus ref's) of -0.001 and ZERO noise:
      LEVEL diff, mean over 8 pairs: -0.0005  (HALF the true effect, and
        individual pairs swing between +0.001 and -0.002 from pair ORDER
        ALONE, with no noise involved at all)
      DELTA diff (this frame's reading minus the PREVIOUS frame's), same
        8 pairs: -0.001  (the FULL true effect, identical on every pair)

Measured on GSLAM/fr1_desk (a KNOWN KEEP scene, offline full-trajectory ATE
2.63->2.85 when halved): the LEVEL-based fallback gave a confident WRONG
REDUCE (t=-2.7, past the significance threshold) - and it REPLICATED
identically on an independent rerun (t=-2.7 again, -9.1% then -10.7%), ruling
out ordinary sampling noise as the explanation. A simulation built to test a
DIFFERENT mechanism (mean-reverting contamination, i.e. AR(1) relaxation
toward a per-k equilibrium - see the now-superseded slot_frames design below)
did NOT reproduce a wrong-answer failure mode from THAT mechanism (only power
loss). The level-vs-delta dilution above is the mechanism that DOES reproduce
it, both algebraically (exactly, as shown) and in a noisy simulation matching
real pair counts: DELTA resolves the correct verdict 3-10x more often than
LEVEL at the SAME budget (6-20 pairs), in both directions, with no increase
in wrong-rate - see utils/test_online_lr_tuner.py.

PoseErrDelta (below) turns a caller's per-frame absolute error level into the
delta OnlineLRTuner should actually receive as pose_err - THIS class changes,
observe()'s statistics do not: it is purely a better-aimed measurement of the
same paired-t machinery already here, not a schedule change. See
Gaussian-SLAM/src/entities/tracker.py for the caller.

SLOT_FRAMES (superseded for the pose_err problem, kept as tested, unused
capability - see git history for the fuller account). widens each role's turn
from one frame to slot_frames CONSECUTIVE frames at one constant k, dropping
the first slot_warmup frames of each block before averaging. It was built on
the AR(1)-relaxation theory above, which simulation did not support - it
recovers POWER under a genuinely mean-reverting contamination process (if one
exists), at 4-8x more frames, but was never shown to fix a wrong ANSWER, which
turned out to need the delta fix above instead, at no extra frame cost.
slot_frames=1, slot_warmup=0 (the default) is exact-equivalent to plain
per-frame interleaving; no existing caller is affected unless it opts in.
"""
import math

DEFAULT_BLOCK_FRAMES = 12
DEFAULT_MAX_HALVINGS = 4
DEFAULT_CAP_LIMIT = 0.5
DEFAULT_SLOT_FRAMES = 1
DEFAULT_SLOT_WARMUP = 0

# Student-t, two-sided 2% level (the 0.99 quantile), df = 1..20. Pocock's
# constant nominal level for up to four interim looks holds the overall
# false-decision rate near 5%. No scipy dependency: this runs inside the
# frontends. Beyond df=20 the normal 0.99 quantile (2.326) is within 12%.
_T_CRIT_02 = (
    31.821, 6.965, 4.541, 3.747, 3.365, 3.143, 2.998, 2.896, 2.821, 2.764,
    2.718, 2.681, 2.650, 2.624, 2.602, 2.583, 2.567, 2.552, 2.539, 2.528)
_T_CRIT_INF = 2.326


def _crit(df):
    if df < 1:
        return math.inf
    return _T_CRIT_02[df - 1] if df <= len(_T_CRIT_02) else _T_CRIT_INF


def _paired_t(diffs):
    """Paired t on the per-pair differences (candidate - reference).
    Negative means the candidate is smaller (REDUCE). Returns (mean, se, t)."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n)
    if se > 0:
        return mean, se, mean / se
    if mean != 0:
        # A constant difference is the STRONGEST evidence, not "no info" -
        # same fix as lr_probe.py's paired().
        return mean, se, (math.inf if mean > 0 else -math.inf)
    return mean, se, 0.0


class PoseErrDelta:
    """Turns a caller's per-frame ABSOLUTE pose-error level into the per-frame
    CHANGE that should actually be passed as OnlineLRTuner's pose_err - see
    the module docstring's POSE_ERR section for why the level itself is a
    diluted, order-confounded estimator under alternating interleaving, and
    the delta recovers the full, unbiased effect at the same frame cost.

    Stateful: call update() once per frame, IN ORDER, whether or not this
    frame's level is itself usable - the reference has to track calendar
    time, not just usable readings, or a gap changes what "the previous
    frame" means silently.
    """

    def __init__(self):
        self._prev = None

    def update(self, level, valid=True):
        """level: this frame's absolute pose error. Returns the delta (level
        minus the last VALID level seen), or None when there is no delta yet
        to report - the first valid frame ever (nothing to difference
        against), or this frame's own level is not usable.

        valid=False (e.g. an invalid ground-truth pose - see
        Gaussian-SLAM/src/entities/tracker.py's gt_pose_valid) means this
        frame's level cannot be trusted AT ALL, not even as a future
        reference: the reference is left unmoved, so the NEXT valid frame's
        delta spans back across the gap to the last frame that was trustworthy
        (a multi-frame delta) rather than silently comparing against a bad
        reading, which would be worse - a stale reference is a smaller, safer
        error than a wrong one.
        """
        if not valid or level is None or not math.isfinite(level):
            return None
        delta = None if self._prev is None else level - self._prev
        self._prev = level
        return delta


class OnlineLRTuner:
    """Drives an acquisition-step scale k from 1.0 down, using only the
    per-frame iteration counts a deployed run already produces.

    Call observe(frame_iters) once per completed frame. Read .k (or
    .scaled(base)) for the value to apply to the active acquisition knob on
    the NEXT frame - it now changes EVERY frame while a rung is open, because
    the two arms are interleaved, and settles once .frozen.  A decision made
    at the end of frame N changes frame N+1 onward, never frame N itself, so
    nothing needs to be undone mid-frame.

    block_frames sets the RAW-FRAME budget of a rung's first look (min_pairs
    ROUNDS, each costing 2*slot_frames raw frames, chosen so 2*min_pairs*
    slot_frames is close to block_frames); a rung is allowed twice that many
    rounds before an undecided result counts as AMBIGUOUS.

    slot_frames/slot_warmup widen each role's turn from a single frame to a
    slot_frames-frame block, dropping the first slot_warmup frames of it from
    that block's reading - see the module docstring's SLOT_FRAMES section for
    why. slot_frames=1 (the default) is unchanged single-frame interleaving.
    """

    def __init__(self, block_frames=DEFAULT_BLOCK_FRAMES,
                 max_halvings=DEFAULT_MAX_HALVINGS,
                 t_threshold=None,
                 cap_limit=DEFAULT_CAP_LIMIT,
                 budget=None, log_fn=None,
                 slot_frames=DEFAULT_SLOT_FRAMES,
                 slot_warmup=DEFAULT_SLOT_WARMUP):
        self.slot_frames = max(1, int(slot_frames))
        self.slot_warmup = int(slot_warmup)
        if self.slot_warmup >= self.slot_frames:
            raise ValueError(
                f"slot_warmup ({self.slot_warmup}) must be < slot_frames "
                f"({self.slot_frames}) - a slot with nothing left after "
                f"warmup has no reading")
        # block_frames stays a RAW-FRAME budget regardless of slot size, so
        # ONLINE_LR_BLOCK's existing meaning does not quietly change cost when
        # a caller also sets slot_frames>1: dividing by slot_frames here keeps
        # the total raw frames of the first look close to what was asked for.
        pairs = max(2, int(block_frames) // (2 * self.slot_frames))
        self.min_pairs = pairs + (pairs % 2)
        self.max_pairs = 2 * self.min_pairs
        self.block_frames = 2 * self.min_pairs * self.slot_frames
        self.max_halvings = int(max_halvings)
        self.t_threshold = None if t_threshold is None else float(t_threshold)
        self.cap_limit = float(cap_limit)
        self.budget = int(budget) if budget else None
        self._log = log_fn or (lambda msg: None)

        self.frozen = False
        self.halvings_done = 0
        self.frames_seen = 0

        # (k_tested, verdict, t, mean_d, n_pairs, mean_ref) per completed rung
        self.history = []
        self.k_log = []   # the k each observed frame ran at, in frame order
        self._ref, self._cand = 1.0, 0.5
        self._start_rung()

    # -- schedule -----------------------------------------------------------
    def _start_rung(self):
        self._diffs = []
        # Parallel to _diffs, one entry per pair: the pose-error difference
        # (candidate - reference), or None if either side of the pair did not
        # report one. Only used when the rung turns out to be cap-bound.
        self._err_diffs = []
        self._rung_frames = 0
        self._rung_capped = 0
        self._rung_cap_known = 0   # frames where at-cap status was determined
        self._ref_sum = 0.0
        self._err_ref_sum = 0.0
        self._open_round()

    def _open_round(self):
        # A ROUND is one slot at the reference k and one at the candidate k.
        # Which leads alternates round to round - same argument as the module
        # docstring's WHICH ARM GOES FIRST, now at slot instead of frame
        # granularity, so a slow scene-difficulty trend still cancels over an
        # even number of rounds regardless of slot_frames.
        self._order = (("ref", "cand") if len(self._diffs) % 2 == 0
                       else ("cand", "ref"))
        self._round_pos = 0
        self._got = {}
        self._open_slot()

    def _open_slot(self):
        self._slot_it = []
        self._slot_err = []
        self.k = self._k_of(self._order[self._round_pos])

    def _k_of(self, role):
        return self._ref if role == "ref" else self._cand

    def scaled(self, base_target_step_norm):
        return base_target_step_norm * self.k

    # -- feed ---------------------------------------------------------------
    def observe(self, frame_iters, at_cap=None, pose_err=None):
        """One completed frame's iteration count. Returns True if .k just
        changed (the caller should re-apply it before the next frame) - with
        slot_frames>1 that is only on the frame AFTER a slot completes, not
        every frame.

        at_cap: whether THIS frame ran its own full budget, if the caller
        knows it directly (e.g. GSLAM's own num_iters, which can vary frame to
        frame). Falls back to frame_iters >= budget when omitted, matching the
        previous behaviour, for callers with a fixed budget.

        pose_err: this frame's committed-pose error against ground truth, if
        the caller has it. Only read when a rung turns out to be cap-bound;
        harmless to always pass.
        """
        self.frames_seen += 1
        if frame_iters > 0:
            # The k this frame RAN at (self.k is only updated below, after the
            # frame is counted) - including frames after the freeze. Aligned
            # with the preconditioner's per-frame series, which are appended
            # under the same frame_iters > 0 gate.
            self.k_log.append(self.k)
        if self.frozen or frame_iters <= 0:
            return False

        before = self.k
        self._rung_frames += 1
        if at_cap is None and self.budget is not None:
            at_cap = frame_iters >= self.budget
        if at_cap is not None:
            self._rung_cap_known += 1
            if at_cap:
                self._rung_capped += 1

        self._slot_it.append(frame_iters)
        self._slot_err.append(pose_err)
        if len(self._slot_it) < self.slot_frames:
            return self.k != before   # mid-slot: k is unchanged (always False)

        # SLOT COMPLETE. Summarise its POST-WARMUP TAIL only - the first
        # slot_warmup frames still ran (spending real time, still counted
        # above) but are dropped here, since they mostly reflect the handoff
        # from whatever the OTHER role just left the trajectory in, not this
        # slot's own k. slot_frames=1, slot_warmup=0 makes tail = the single
        # frame just observed, identical to the pre-slot behaviour.
        role = self._order[self._round_pos]
        tail_it = self._slot_it[self.slot_warmup:]
        tail_err = [e for e in self._slot_err[self.slot_warmup:]
                   if e is not None and math.isfinite(e)]
        self._got[role] = (sum(tail_it) / len(tail_it),
                           sum(tail_err) / len(tail_err) if tail_err else None)

        self._round_pos += 1
        if self._round_pos < 2:
            self._open_slot()
            return True   # k changes to the other role's for the next frame

        (ref_it, ref_err), (cand_it, cand_err) = self._got["ref"], self._got["cand"]
        self._diffs.append(cand_it - ref_it)
        self._ref_sum += ref_it
        # NaN is not None, and would sail past a bare "is not None" check and
        # silently poison the whole rung's mean/t-stat (sum/mean of ANY
        # sequence containing a NaN is NaN). Guard it here, not just at each
        # caller - a ground truth pose can be NaN by construction (see
        # Gaussian-SLAM/src/entities/tracker.py's ScanNet gt_pose_valid), and
        # a future caller passing pose_err should not have to remember this.
        # (Also redundant with the tail_err filter above, which already drops
        # non-finite values per FRAME before they reach a slot mean - kept
        # here too since a slot mean is itself always finite once computed.)
        if (ref_err is not None and cand_err is not None
                and math.isfinite(ref_err) and math.isfinite(cand_err)):
            self._err_diffs.append(cand_err - ref_err)
            self._err_ref_sum += ref_err
        else:
            self._err_diffs.append(None)
        n = len(self._diffs)
        if n >= self.min_pairs and n % 2 == 0:
            self._decide()
        else:
            self._open_round()
        return self.k != before or self.frozen

    # -- decisions ----------------------------------------------------------
    def _freeze(self, k, reason):
        self.frozen = True
        self.k = k
        self._log(f"[OnlineLRTuner] frozen at k={k:.4g} after "
                  f"{self.frames_seen} frames: {reason}")

    def _decide(self):
        # n: ROUNDS RUN this rung - governs SCHEDULING only (min_pairs,
        # max_pairs, whether to keep collecting) and never changes below.
        # n_stat: the sample size the ACTIVE signal's t-test actually used -
        # equals n for it/frame, but for pose_err can be n-1 (the tolerated
        # bootstrap hole, see below) and must drive crit()'s degrees of
        # freedom and everything reported, or a 5-sample statistic would be
        # logged and judged as if it were a 6-sample one.
        n = n_stat = len(self._diffs)
        mean_d, se, t = _paired_t(self._diffs)
        mean_ref = self._ref_sum / n
        capped_frac = (self._rung_capped / self._rung_cap_known
                      if self._rung_cap_known > 0 else 0.0)
        ref, cand = self._ref, self._cand
        signal = "it/frame"

        def record(verdict):
            self.history.append((cand, verdict, t, mean_d, n_stat, mean_ref, signal))

        if self._rung_cap_known > 0 and capped_frac >= self.cap_limit:
            # Checked before the t-test: a pair pinned in both arms differs by
            # exactly zero, so the comparison is diluted, not just noisy.
            valid_err = [d for d in self._err_diffs if d is not None]
            # TOLERATE AT MOST ONE MISSING PAIR, not zero. A caller feeding
            # PoseErrDelta (the recommended path - see the module docstring's
            # POSE_ERR section) has EXACTLY one unavoidable hole: its very
            # first call, ever, for the whole run has no previous reading to
            # difference against and returns None by construction - not a
            # data problem, a one-time bootstrap cost paid once per run, never
            # again (later rungs, even after a k halving, inherit an already-
            # primed reference). Requiring every pair - the original,
            # stricter rule - meant the fallback could NEVER activate on a
            # run's first rung regardless of the true effect size, which is
            # what happened on GSLAM/fr1_desk and room0 (2026-09-22): both
            # declined at "9/10 pairs", not because the data was thin, but
            # because THIS check made a guaranteed single hole look like a
            # data failure. A second missing pair still declines - that one
            # is unexplained by the bootstrap and IS worth distrusting.
            if len(valid_err) >= n - 1:   # n >= min_pairs >= 2, so this implies >= 1
                # Enough pairs have a reading - fall back to pose error,
                # exactly what lr_probe.py's auto signal does offline. Same
                # sign convention as it/frame (mean_d < 0 -> REDUCE), so the
                # decision branches below need no change.
                signal = "pose_err"
                n_stat = len(valid_err)
                mean_d, se, t = _paired_t(valid_err)
                mean_ref = self._err_ref_sum / n_stat
            else:
                record("CAP-BOUND")
                self._freeze(ref,
                             f"{100 * capped_frac:.0f}% of this rung's frames "
                             f"hit the iteration cap - it/frame has no power "
                             f"here (same as lr_probe.py's cap-bound finding), "
                             f"and only {len(valid_err)}/{n} pairs have a "
                             f"pose-error reading to fall back on (more than "
                             f"the one bootstrap hole tolerated); keeping "
                             f"k={ref:.4g} rather than trust a diluted "
                             f"comparison")
                return

        crit = self.t_threshold if self.t_threshold is not None \
            else _crit(n_stat - 1)
        pct = 100 * mean_d / mean_ref if mean_ref else 0.0
        sig_tag = "" if signal == "it/frame" else f" [{signal}]"

        if abs(t) < crit:
            if n < self.max_pairs:   # SCHEDULING: rounds run, not n_stat
                self._open_round()   # undecided: take two more rounds
                return
            record("AMBIGUOUS")
            self._freeze(ref,
                         f"k={cand:.4g} vs {ref:.4g} was AMBIGUOUS after "
                         f"{n_stat} pairs (t={t:+.1f} < {crit:.2f}, "
                         f"{pct:+.1f}%){sig_tag} - keeping the larger, safer "
                         f"value")
            return

        if t > 0:
            # Frames got LONGER (or the pose ended FURTHER from ground truth)
            # at the smaller k - it needs the step it had.
            record("KEEP")
            self._freeze(ref,
                         f"k={cand:.4g} was worse than k={ref:.4g} "
                         f"({pct:+.1f}%, t={t:+.1f}, {n_stat} pairs){sig_tag}")
            return

        # t < 0: shorter frames, or a pose closer to ground truth - accept and
        # keep halving if budget allows.
        record("REDUCE")
        self.halvings_done += 1
        if self.halvings_done >= self.max_halvings:
            self._freeze(cand,
                         f"reached max_halvings={self.max_halvings} while "
                         f"still seeing REDUCE - freezing at the smallest "
                         f"value tried instead of halving indefinitely")
            return
        self._log(f"[OnlineLRTuner] k={cand:.4g} beats k={ref:.4g} "
                  f"({pct:+.1f}%, t={t:+.1f}, {n_stat} pairs){sig_tag} - trying "
                  f"k={cand / 2:.4g}")
        self._ref, self._cand = cand, cand / 2.0
        self._start_rung()

    @property
    def decided_k(self):
        """The k to ship. Frozen: the frozen value. Still mid-rung (a run cut
        short, e.g. the single-run ladder's frame budget): the last ACCEPTED
        k, never the candidate being tested - an undecided rung is not a
        REDUCE, by the same asymmetry that makes AMBIGUOUS keep the larger."""
        return self.k if self.frozen else self._ref

    def summary(self):
        head = (f"OnlineLRTuner: frozen={self.frozen} "
                f"decided_k={self.decided_k:.4g} after {self.frames_seen} "
                f"frames, {self.halvings_done} halving(s) accepted")
        if not self.history:
            return head + " (no rung completed)"
        lines = [head]
        for k_tested, verdict, t, mean_d, n, mean_ref, signal in self.history:
            pct = 100 * mean_d / mean_ref if mean_ref else 0.0
            tag = "" if signal == "it/frame" else f"  [{signal}]"
            # mean_delta uses %g, not %.2f: pose_err deltas are ~1e-4..1e-3,
            # and fixed 2-decimal formatting rounded every one of them to
            # "-0.00" - correct in the decision (which uses full-precision
            # mean_d/t) but unreadable in the log. it/frame deltas still print
            # the same as before (%.4g of e.g. -21.2 is "-21.2").
            lines.append(f"  k={k_tested:.4g}  {verdict:10s} "
                         f"t={t:+.1f}  mean_delta={mean_d:+.4g} ({pct:+.1f}%)"
                         f"  pairs={n}{tag}")
        return "\n".join(lines)
