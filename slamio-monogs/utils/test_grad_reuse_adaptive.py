"""Adaptive cooldown: the boundary must move on evidence, not on an index.

WHY THIS EXISTS. Every fixed-cooldown arm measured on GSLAM failed for the
same reason: frames run 84-200 iterations and staleness tracks how close a
frame is to converging, not its absolute iteration count, so one global index
is wrong for most frames. Adaptive mode replaces the index with a boundary
that EXTENDS on a trusted check and FREEZES on a failed one, driven by
cos(g_fresh_now, g_fresh_previous) - the cheapest signal available without
computing the very gradient reuse is trying to skip.

The two properties that matter, and that a short manual test would not catch:

  1. THE SYNC IS BATCHED. cos is computed on-device every fresh call;
     .item() runs only once every `check_every` calls. A bug here silently
     turns a batched design back into a per-iteration one - exactly the
     cost this file's module docstring says was measured at ~24% of
     tracking wall time in an earlier attempt.

  2. A FAILED CHECK FREEZES, IT DOES NOT SHRINK. Iterations already inside
     an extended boundary keep reusing; only further extension stops. This
     is the fail-safe property: a spurious low reading (the it20 DIAG-switch
     artefact that prompted this design) ends reuse a little early for one
     frame rather than retroactively invalidating iterations already run.

Runs on CPU - everything here is arithmetic on small tensors, no CUDA needed.

    python utils/test_grad_reuse_adaptive.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grad_reuse import GradReuse  # noqa: E402

_fails = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          "" if not detail else "  (%s)" % detail))
    if not ok:
        _fails.append(name)


def _gr(**over):
    cfg = dict(enabled=True, period=2, warmup=4, adaptive=True,
              trust_cos=0.85, check_every=2)
    cfg.update(over)
    return GradReuse(cfg)


def _run(gr, grads, warmup_stashes=True):
    """Simulate a frame: grads[i] is the FRESH gradient at even iterations,
    None at odd (reuse-candidate) iterations where the closure would just
    read the stash instead. Returns the should_reuse() decision series.
    """
    gr.reset_frame()
    decisions = []
    for i, g in enumerate(grads):
        is_last = i == len(grads) - 1
        reused = gr.should_reuse(i, is_last)
        decisions.append(reused)
        if not reused:
            # A render iteration: feed the trust signal, then stash.
            gr.note_fresh_grad(g, i)
            gr.note_rendered()
        else:
            gr.note_reused()
    return decisions


# --- independent bootstrap hold does not fit or mutate a cooldown ----------
gr_hold = _gr(hold_frames=2)
_hold_grad = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
hold_frame1 = _run(gr_hold, [_hold_grad] * 20)
hold_frame2 = _run(gr_hold, [_hold_grad] * 20)
hold_frame3 = _run(gr_hold, [_hold_grad] * 20)
check("bootstrap hold disables every reuse decision in its frame prefix",
      not any(hold_frame1) and not any(hold_frame2))
check("bootstrap hold releases reuse on the following frame",
      any(hold_frame3), "frame3=%s" % hold_frame3)
check("bootstrap hold does not fit or change the cooldown",
      gr_hold.cooldown == 0 and gr_hold.calibration_frames == 0,
      "cooldown=%s calibration_frames=%s" %
      (gr_hold.cooldown, gr_hold.calibration_frames))


# --- default (non-adaptive) path is byte-identical to before ---------------
gr_default = GradReuse(dict(enabled=True, period=2, warmup=4, cooldown=20))
gr_adaptive_off = GradReuse(dict(enabled=True, period=2, warmup=4, cooldown=20,
                                 adaptive=False))
d1 = _run(gr_default, [torch.randn(6) for _ in range(30)])
d2 = _run(gr_adaptive_off, [torch.randn(6) for _ in range(30)])
check("adaptive=False leaves should_reuse() behaviour on the index alone",
      d1 == d2)


# --- a stable gradient direction keeps extending the boundary --------------
STABLE = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
gr = _gr(warmup=4, check_every=2)
grads = [STABLE * (1.0 + 0.01 * i) for i in range(60)]  # same direction, drifting magnitude
decisions = _run(gr, grads)
reused_past_40 = any(decisions[40:])
check("a stable gradient direction lets reuse extend past iteration 40",
      reused_past_40,
      "reused at %s" % [i for i, r in enumerate(decisions) if r and i >= 40][:5])
check("boundary actually moved past its initial grace window",
      gr._boundary > gr.warmup + gr.check_every * gr.period,
      "boundary=%s initial=%s" % (gr._boundary,
                                  gr.warmup + gr.check_every * gr.period))


# --- an abrupt direction flip (the it20-restart shape) freezes it ----------
gr2 = _gr(warmup=4, check_every=2)
grads2 = ([STABLE] * 20
         + [torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])] * 4  # the flip
         + [STABLE] * 20)
decisions2 = _run(gr2, grads2)
boundary_at_flip = gr2._boundary
# After the flip is observed, no MORE extension should occur, even though
# the gradient later returns to STABLE - the freeze is not self-healing by
# design (a fresh frame gets a fresh boundary; this one does not recover
# mid-frame from what looked like a bad read).
check("a sign flip freezes the boundary rather than growing it further",
      gr2._boundary <= boundary_at_flip + gr2.check_every * gr2.period,
      "boundary=%s at-flip=%s" % (gr2._boundary, boundary_at_flip))
check("some checks were recorded as frozen",
      gr2.adaptive_freezes > 0, "freezes=%d" % gr2.adaptive_freezes)


# --- local_trust: the SAME flip-then-recover sequence, but reuse resumes ---
# This is the opposite property from gr2 above, on purpose: local_trust
# drops the ratchet, so a transient flip stops reuse only while it is
# actually failing checks - once the gradient recovers to STABLE, a later
# check passes again and reuse turns back on, instead of staying frozen for
# the rest of the frame.
#
# EXACT INDICES EMPIRICALLY VERIFIED, not hand-derived: once a check fails,
# should_reuse() returns False for the NEXT odd iteration too, which means
# IT also renders (note_fresh_grad runs on every non-reused iteration, not
# only the mandatory even ones) - so the fresh-grad sequence goes from
# every-other to every-single-iteration the moment trust drops, which
# speeds up how fast a recovering signal is detected. Iteration 20 is the
# flip's first element; 23 and 25 are confirmed-dropped, 27 confirmed-
# resumed for this exact warmup=4/check_every=2/period=2 configuration -
# re-derive with utils/grad_reuse.py's note_fresh_grad logic directly
# (not by hand) if these parameters ever change.
gr17 = _gr(warmup=4, check_every=2, local_trust=True)
decisions17 = _run(gr17, grads2)  # same [STABLE]*20 + [FLIPPED]*4 + [STABLE]*20
check("local_trust: reuse actually stops once the flip is detected (it23, it25)",
      not decisions17[23] and not decisions17[25],
      "it23=%s it25=%s" % (decisions17[23], decisions17[25]))
check("local_trust: reuse RESUMES after recovering to STABLE (it27, not a ratchet)",
      decisions17[27],
      "it27=%s" % decisions17[27])
check("local_trust: no _boundary is tracked (nothing to report)",
      gr17._boundary is None, "boundary=%s" % gr17._boundary)


# --- a frozen boundary does not retroactively cancel earlier reuse ---------
# Iterations already inside the boundary when the freeze happens must still
# have been allowed to reuse - check the decisions list directly rather than
# the final state, since should_reuse() is called forward-only in the real
# loop and never revisits a past iteration.
check("iterations before the freeze point were still allowed to reuse",
      any(decisions2[6:20]),
      "decisions[6:20]=%s" % decisions2[6:20])


# --- cooldown acts as a ceiling in adaptive mode, not a switch-off ---------
gr3 = _gr(warmup=4, check_every=2, cooldown=30)
grads3 = [STABLE for _ in range(80)]
decisions3 = _run(gr3, grads3)
check("cooldown caps the adaptive boundary even under perfect trust",
      gr3._boundary <= 30, "boundary=%s" % gr3._boundary)
check("nothing reuses at or past the cooldown ceiling",
      not any(decisions3[30:]),
      "reused past 30 at %s" % [i for i, r in enumerate(decisions3) if r and i >= 30])
checks_at_ceiling = gr3.adaptive_checks
gr3.note_fresh_grad(STABLE, gr3.cooldown)
check("the adaptive checker itself is off past the cooldown ceiling",
      gr3.adaptive_checks == checks_at_ceiling,
      "before=%d after=%d" % (checks_at_ceiling, gr3.adaptive_checks))


# --- the sync is genuinely batched: cos is computed every call, --------
# .item() only at check boundaries -------------------------------------
class _CountingTensor:
    """Wraps a real tensor and counts .item() calls, to catch a batching
    regression that a value-only test would miss."""
    item_calls = [0]

    def __init__(self, t):
        self._t = t

    def __getattr__(self, name):
        attr = getattr(self._t, name)
        return attr


gr4 = _gr(warmup=0, check_every=5)
gr4.reset_frame()
_orig_item = torch.Tensor.item
calls = [0]


def _counting_item(self):
    calls[0] += 1
    return _orig_item(self)


torch.Tensor.item = _counting_item
try:
    for i in range(20):
        gr4.note_fresh_grad(STABLE.clone(), i)
finally:
    torch.Tensor.item = _orig_item
# 20 fresh calls, check_every=5 -> checks at calls 5,10,15,20 = 4 syncs,
# not 20. (First call never checks: no previous gradient to compare.)
check("item() is called only at check boundaries, not every fresh grad",
      calls[0] == 4, "item() calls=%d, expected 4" % calls[0])


# --- g=None is treated as no evidence, never a crash or a forced freeze ----
gr5 = _gr(warmup=0, check_every=2)
gr5.reset_frame()
try:
    gr5.note_fresh_grad(None, 0)
    gr5.note_fresh_grad(STABLE, 1)
    gr5.note_fresh_grad(None, 2)
    ok = True
except Exception as exc:  # noqa: BLE001
    ok = False
    print("    raised: %r" % exc)
check("note_fresh_grad(None, ...) is a safe no-op", ok)


# --- run-in calibration disables reuse, then selects a rounded ceiling -----
gr6 = _gr(
    warmup=20,
    cooldown=0,
    calibration_frames=5,
    calibration_margin=15,
    calibration_round=10,
)
calibration_blocked = []
for stop, reason in zip(
        [68, 70, 72, 74, 70],
        ["native", "window", "native", "cap", "window"]):
    gr6.reset_frame()
    gr6.note_rendered()
    calibration_blocked.append(not gr6.should_reuse(21, False))
    gr6.note_frame_stop(stop, reason)
check("reuse is disabled throughout all calibration frames",
      all(calibration_blocked), "blocked=%s" % calibration_blocked)
check("median 70 minus 15 rounded down to a decade selects cooldown 50",
      gr6.cooldown == 50,
      "median=%s raw=%s selected=%s" % (
          gr6._calibration_median,
          gr6._calibration_raw_cutoff,
          gr6.cooldown,
      ))
gr6.reset_frame()
gr6.note_rendered()
check("reuse opens after calibration inside the selected window",
      gr6.should_reuse(21, False))
check("the selected cooldown remains a hard adaptive ceiling",
      not gr6.should_reuse(51, False))


# If the observed stop is too close to DIAG, preserve one complete decade of
# usable window instead of silently deriving cooldown <= warmup (an inert run).
gr7 = _gr(
    warmup=20,
    cooldown=0,
    calibration_frames=3,
    calibration_margin=15,
    calibration_round=10,
)
for stop in [24, 25, 26]:
    gr7.reset_frame()
    gr7.note_frame_stop(stop, "native")
check("an early median clamps to one decade beyond warmup",
      gr7.cooldown == 30,
      "raw=%s selected=%s" % (gr7._calibration_raw_cutoff, gr7.cooldown))


# GSLAM's closed-loop guard intentionally sends early calibration frames to
# the cap. They describe guard bookkeeping, not where the stopper fires, so
# stopping-only calibration filters them while retaining a 50-frame run-in.
gr8 = _gr(
    warmup=40,
    cooldown=0,
    calibration_frames=5,
    calibration_margin=15,
    calibration_round=10,
    calibration_stat="min",
    calibration_exclude_reasons=("cap",),
)
for stop, reason in [
        (200, "cap"), (200, "cap"),
        (96, "stopping"), (120, "stopping"), (136, "stopping")]:
    gr8.reset_frame()
    gr8.note_frame_stop(stop, reason)
check("cap frames are excluded from GSLAM's stopping statistic",
      gr8._calibration_stops == [96, 120, 136],
      "eligible=%s" % gr8._calibration_stops)
check("earliest actual stop 96 selects conservative cooldown 80",
      gr8.cooldown == 80,
      "anchor=%s selected=%s" % (gr8._calibration_anchor, gr8.cooldown))


gr9 = _gr(
    warmup=40,
    calibration_frames=2,
    calibration_exclude_reasons=("cap",),
)
for _ in range(2):
    gr9.reset_frame()
    gr9.note_frame_stop(200, "cap")
gr9.reset_frame()
gr9.note_rendered()
check("no actual stops leaves reuse safely disabled",
      not gr9.should_reuse(41, False))


# --- short-horizon calibration keeps reuse alive on SplaTAM-sized frames --
# DIAG=20 against a stop at 25-28 leaves fewer than two complete adaptive
# check horizons (2 * check_every * period = 16).  The opt-in fallback selects
# a period-aligned fixed window instead of pretending the long-frame cosine
# gate has enough observations to adapt.
def _short_calibration(stop):
    gr = _gr(
        warmup=20,
        cooldown=0,
        check_every=4,
        calibration_frames=5,
        calibration_stat="min",
        calibration_exclude_reasons=("cap",),
        calibration_short_horizon=True,
        calibration_phase_boundary=20,
    )
    blocked = []
    for _ in range(5):
        gr.reset_frame()
        gr.note_rendered()
        blocked.append(not gr.should_reuse(21, False))
        gr.note_frame_stop(stop, "stopping")
    return gr, blocked


gr10, blocked10 = _short_calibration(25)
check("short-horizon calibration blocks reuse throughout its run-in",
      all(blocked10), "blocked=%s" % blocked10)
check("SplaTAM stop 25 selects the measured w6/c20 window",
      (gr10.warmup, gr10.cooldown) == (6, 20),
      "selected=%s/%s" % (gr10.warmup, gr10.cooldown))
gr10.reset_frame()
gr10.note_rendered()
check("the calibrated SplaTAM window reuses inside w6/c20",
      gr10.should_reuse(7, False) and not gr10.should_reuse(20, False))
check("short mode is fixed even when adaptive was requested",
      gr10._fixed_calibrated_window and gr10._boundary is None)


gr11, _ = _short_calibration(28)
check("SplaTAM stop 28 selects a period-aligned w6/c22 window",
      (gr11.warmup, gr11.cooldown) == (6, 22),
      "selected=%s/%s" % (gr11.warmup, gr11.cooldown))


# The fallback is horizon-triggered, not a SplaTAM name check.  A long frame
# keeps both its configured warmup and the existing minus-15/round-10 rule.
gr12 = _gr(
    warmup=40,
    cooldown=0,
    calibration_frames=3,
    calibration_margin=15,
    calibration_round=10,
    calibration_stat="min",
    calibration_short_horizon=True,
    calibration_phase_boundary=40,
)
for stop in [96, 120, 136]:
    gr12.reset_frame()
    gr12.note_frame_stop(stop, "stopping")
check("long frames retain the existing calibration rule unchanged",
      (gr12.warmup, gr12.cooldown, gr12._fixed_calibrated_window)
      == (40, 80, False),
      "selected=%s/%s fixed=%s" % (
          gr12.warmup, gr12.cooldown, gr12._fixed_calibrated_window))


# --- trust_mag: a magnitude gate ALONGSIDE cosine, not instead of it -------
# Perfect direction throughout (cos=1 every pair) but magnitude swings hard
# between 1x and 5x every other fresh gradient - a case trust_cos alone
# cannot see at all.
SWING = [STABLE * (5.0 if (i // 2) % 2 else 1.0) for i in range(60)]

gr13 = _gr(warmup=4, check_every=2, trust_mag_enabled=False)
d13 = _run(gr13, SWING)
check("trust_mag=off: magnitude swings do not stop extension (cos-only baseline)",
      any(d13[40:]), "reused past 40: %s" % any(d13[40:]))

gr14 = _gr(warmup=4, check_every=2, trust_mag_enabled=True,
          trust_mag_low=0.8, trust_mag_high=1.2)
d14 = _run(gr14, SWING)
check("trust_mag=on: the SAME magnitude swing now freezes the boundary early",
      not any(d14[40:]), "reused past 40: %s" % any(d14[40:]))
check("trust_mag=on: freezes are attributed to the magnitude gate",
      gr14.adaptive_freezes_mag > 0,
      "freezes=%d freezes_mag=%d" % (gr14.adaptive_freezes,
                                     gr14.adaptive_freezes_mag))

# --- trust_mag does not fire when cosine ALREADY rejected the pair ---------
# Both gates fail together (direction flips AND magnitude swings) - the
# freeze must count once, attributed to cos, not double-counted as a
# magnitude freeze too (cos_trusted is already False).
BOTH_BAD = [STABLE * (1.0 if (i // 2) % 2 == 0 else -5.0) for i in range(30)]
gr15 = _gr(warmup=4, check_every=2, trust_mag_enabled=True,
          trust_mag_low=0.8, trust_mag_high=1.2)
_run(gr15, BOTH_BAD)
check("trust_mag: a pair failing BOTH gates is not double-counted as mag-only",
      gr15.adaptive_freezes_mag == 0 and gr15.adaptive_freezes > 0,
      "freezes=%d freezes_mag=%d" % (gr15.adaptive_freezes,
                                     gr15.adaptive_freezes_mag))

# --- widening the band accepts the same swing that a tight band froze -----
gr16 = _gr(warmup=4, check_every=2, trust_mag_enabled=True,
          trust_mag_low=0.1, trust_mag_high=10.0)
d16 = _run(gr16, SWING)
check("trust_mag: a band wide enough to cover the swing behaves like it is off",
      any(d16[40:]), "reused past 40: %s" % any(d16[40:]))

try:
    _gr(trust_mag_enabled=True, trust_mag_low=1.2, trust_mag_high=0.8)
    _raised = False
except ValueError:
    _raised = True
check("trust_mag_low must be < trust_mag_high", _raised)


# --- rejection circuit breaker: persistent distrust becomes fresh-only ----
gr_break = _gr(warmup=8, check_every=1, local_trust=True,
               reject_patience=3)
FLIPPED = -STABLE
alternating = [STABLE if i % 2 == 0 else FLIPPED for i in range(16)]
decisions_break = _run(gr_break, alternating)
checks_at_latch = gr_break.adaptive_checks
check("three consecutive rejections latch the frame into fresh-only mode",
      gr_break._reject_latched and gr_break._reject_streak == 3,
      "latched=%s streak=%d" %
      (gr_break._reject_latched, gr_break._reject_streak))
check("the rejection latch stops reuse for the rest of that frame",
      not any(decisions_break[4:]),
      "reused=%s" % [i for i, reuse in enumerate(decisions_break) if reuse])
gr_break.note_fresh_grad(STABLE, 100)
check("the rejection latch also stops further trust checks",
      gr_break.adaptive_checks == checks_at_latch,
      "before=%d after=%d" %
      (checks_at_latch, gr_break.adaptive_checks))
decisions_next = _run(gr_break, [STABLE] * 12)
check("the rejection circuit resets and reuse can refire next frame",
      not gr_break._reject_latched and any(decisions_next),
      "latched=%s reused=%s" %
      (gr_break._reject_latched, [i for i, x in enumerate(decisions_next) if x]))


# --- rejection backoff: skip known-bad reads, then probe recovery ---------
gr_backoff = _gr(warmup=0, check_every=1, local_trust=True,
                 reject_backoff=4)
gr_backoff.reset_frame()
gr_backoff.note_fresh_grad(STABLE, 0)
gr_backoff.note_rendered()
gr_backoff.note_fresh_grad(FLIPPED, 1)
gr_backoff.note_rendered()
check("a rejected check starts a real-iteration recovery backoff",
      (gr_backoff.adaptive_checks == 1
       and not gr_backoff._locally_trusted
       and gr_backoff._reject_backoff_until == 5),
      "checks=%d trusted=%s until=%d" %
      (gr_backoff.adaptive_checks, gr_backoff._locally_trusted,
       gr_backoff._reject_backoff_until))
for i in (2, 3, 4):
    check("backoff keeps reuse off at iter %d" % i,
          not gr_backoff.should_reuse(i, False))
    gr_backoff.note_fresh_grad(STABLE, i)
    gr_backoff.note_rendered()
check("backoff suppresses checks while refreshing the gradient baseline",
      (gr_backoff.adaptive_checks == 1
       and gr_backoff.reject_backoff_skips == 3
       and torch.equal(gr_backoff._last_fresh_grad, STABLE)),
      "checks=%d skipped=%d" %
      (gr_backoff.adaptive_checks, gr_backoff.reject_backoff_skips))
check("reuse is still off before the recovery probe",
      not gr_backoff.should_reuse(5, False))
gr_backoff.note_fresh_grad(STABLE, 5)
gr_backoff.note_rendered()
check("the first successful recovery probe re-enables reuse",
      (gr_backoff.adaptive_checks == 2
       and gr_backoff._locally_trusted
       and gr_backoff._reject_backoff_until == 0
       and gr_backoff.should_reuse(7, False)),
      "checks=%d trusted=%s until=%d" %
      (gr_backoff.adaptive_checks, gr_backoff._locally_trusted,
       gr_backoff._reject_backoff_until))

for bad_cfg, label in (
        (dict(local_trust=False, reject_backoff=4), "local trust"),
        (dict(local_trust=True, reject_backoff=4, async_check=True),
         "synchronous checks")):
    try:
        _gr(**bad_cfg)
        _raised = False
    except ValueError:
        _raised = True
    check("reject backoff requires %s" % label, _raised)


# --- batched trust lease: exact drain result, never consecutive reuse -------
gr_batch = _gr(warmup=0, check_every=1, local_trust=True,
               batched_check=True, trust_lease=2)
gr_batch.reset_frame()
check("batched trust starts fresh-only before the first completed drain",
      not gr_batch.should_reuse(0, False))
sig0 = gr_batch.note_fresh_grad(STABLE, 0)
gr_batch.note_rendered()
sig1 = gr_batch.note_fresh_grad(STABLE * 1.01, 1)
gr_batch.note_rendered()
check("batched mode stages a device signal without applying a host decision",
      sig0 is None and sig1 is not None
      and tuple(sig1.shape) == (1,) and gr_batch.adaptive_checks == 0,
      "signal=%s checks=%d" % (sig1, gr_batch.adaptive_checks))
gr_batch_no_sync = _gr(warmup=0, check_every=1, local_trust=True,
                       batched_check=True, trust_lease=2)
gr_batch_no_sync.reset_frame()
_orig_item = torch.Tensor.item
batch_item_calls = [0]


def _count_batched_item(self):
    batch_item_calls[0] += 1
    return _orig_item(self)


torch.Tensor.item = _count_batched_item
try:
    gr_batch_no_sync.note_fresh_grad(STABLE, 0)
    gr_batch_no_sync.note_fresh_grad(STABLE * 1.01, 1)
finally:
    torch.Tensor.item = _orig_item
check("batched signal production performs no standalone item() readback",
      batch_item_calls[0] == 0,
      "item calls=%d" % batch_item_calls[0])
_cos = sig1.tolist()[0]
gr_batch.note_batched_check(_cos, None, None, 1)
check("one trusted drain grants exactly two reuse credits",
      gr_batch._trust_lease_remaining == 2,
      "lease=%d" % gr_batch._trust_lease_remaining)

batch_decisions = []
for i in range(2, 7):
    reused = gr_batch.should_reuse(i, False)
    batch_decisions.append(reused)
    if reused:
        gr_batch.note_reused()
    else:
        gr_batch.note_fresh_grad(STABLE * (1.0 + 0.01 * i), i)
        gr_batch.note_rendered()
check("lease=2 means F,R,F,R and never two consecutive reuses",
      batch_decisions == [False, True, False, True, False]
      and not any(a and b for a, b in zip(batch_decisions,
                                          batch_decisions[1:])),
      "decisions=%s" % batch_decisions)
check("both credits are consumed and reuse then fails closed",
      gr_batch._trust_lease_remaining == 0
      and not gr_batch.should_reuse(7, False))

# A shadow control renders the eligible iteration but must consume the same
# authority as the applying arm.  This is the invariant used by SplaTAM's
# cumulative speedup breakdown when gradient application is disabled.
gr_shadow = _gr(warmup=0, check_every=1, local_trust=True,
                batched_check=True, trust_lease=2)
gr_shadow.reset_frame()
gr_shadow.note_fresh_grad(STABLE, 0)
gr_shadow.note_rendered()
shadow_signal = gr_shadow.note_fresh_grad(STABLE * 1.01, 1)
gr_shadow.note_rendered()
gr_shadow.note_batched_check(shadow_signal.tolist()[0], None, None, 1)
check("shadow control starts with the same two trust credits",
      gr_shadow._trust_lease_remaining == 2)
check("shadow candidate is eligible before its forced render",
      gr_shadow.should_reuse(3, False))
gr_shadow.note_rendered()
gr_shadow.note_reuse_candidate()
check("shadow candidate consumes one lease without claiming a reuse",
      gr_shadow._trust_lease_remaining == 1
      and gr_shadow.reused == 0 and gr_shadow.rendered == 3,
      "lease=%d reused=%d rendered=%d" %
      (gr_shadow._trust_lease_remaining, gr_shadow.reused,
       gr_shadow.rendered))
gr_batch.note_batched_check(-1.0, float("nan"), float("nan"), 7)
check("a rejected drain revokes the lease",
      gr_batch._trust_lease_remaining == 0
      and not gr_batch._locally_trusted)
gr_batch.reset_frame()
check("a frame boundary clears all lease authority",
      gr_batch._trust_lease_remaining == 0)

gr_batch_mag = _gr(warmup=0, check_every=1, local_trust=True,
                   batched_check=True, trust_lease=2,
                   trust_mag_enabled=True, trust_mag_low=0.8,
                   trust_mag_high=1.2)
gr_batch_mag.reset_frame()
gr_batch_mag.note_fresh_grad(STABLE, 0)
mag_signal = gr_batch_mag.note_fresh_grad(STABLE * 1.01, 1)
check("batched trust carries cosine and magnitude in the same drain row",
      mag_signal is not None and tuple(mag_signal.shape) == (3,))
gr_batch_mag.note_batched_check(*mag_signal.tolist(), 1)
check("a trusted batched magnitude reading grants the same lease",
      gr_batch_mag._trust_lease_remaining == 2)

for bad_cfg, label in (
        (dict(local_trust=False, batched_check=True), "local trust"),
        (dict(local_trust=True, batched_check=True, async_check=True),
         "non-async readback"),
        (dict(local_trust=True, batched_check=True, period=3,
              allow_long_period=True), "period two")):
    try:
        _gr(**bad_cfg)
        _raised = False
    except ValueError:
        _raised = True
    check("batched trust requires %s" % label, _raised)


# --- async_check: hidden-sync control flow ----------------------------
# torch.cuda.Event() and pin_memory=True both hard-require a real CUDA
# context - this file's whole point is running on CPU, so both are
# monkeypatched here for the duration of this block. These tests cover
# the CONTROL FLOW (issue once, never double-issue, apply only once
# landed, discard across a frame boundary) - NOT real async overlap or
# timing, which only a real GPU can exercise. Treat a pass here as "the
# plumbing is wired correctly", not "the hidden-sync run was validated
# end-to-end" - that still needs an actual run on the target GPU.
class _FakeEvent:
    """query() returns whatever .ready was last set to - the test drives
    it directly, standing in for 'has the async copy actually landed'."""
    created = []

    def __init__(self):
        self.ready = False
        _FakeEvent.created.append(self)

    def record(self):
        pass

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


_orig_event = torch.cuda.Event
_orig_empty = torch.empty


def _empty_no_pin(*a, **kw):
    # pin_memory=True raises outright without CUDA ("Need to provide pin
    # memory allocator") - drop it so the SAME call shape used in
    # production (torch.empty(shape, dtype=..., device='cpu',
    # pin_memory=True)) runs on this CPU-only test build too.
    kw.pop("pin_memory", None)
    return _orig_empty(*a, **kw)


torch.cuda.Event = _FakeEvent
torch.empty = _empty_no_pin
try:
    # Equivalence: async_check with an event that is ready IMMEDIATELY
    # (zero simulated latency) must produce the IDENTICAL decision
    # sequence as the synchronous path on the same gradient trace - this
    # is the real correctness claim (the async plumbing changes WHEN a
    # decision is consumed, never what it decides from the same data).
    gr_sync = _gr(warmup=4, check_every=2)
    gr_async = _gr(warmup=4, check_every=2, async_check=True)
    STABLE60 = [STABLE * (1.0 + 0.01 * i) for i in range(60)]
    d_sync = _run(gr_sync, STABLE60)

    class _AlwaysReadyEvent(_FakeEvent):
        def __init__(self):
            super().__init__()
            self.ready = True

    torch.cuda.Event = _AlwaysReadyEvent
    d_async = _run(gr_async, STABLE60)
    torch.cuda.Event = _FakeEvent
    check("async_check with zero latency decides identically to the sync path",
          d_sync == d_async,
          "diverged at %s" % [i for i in range(len(d_sync))
                              if d_sync[i] != d_async[i]][:5])
    check("async_check with zero latency reaches the same final boundary",
          gr_sync._boundary == gr_async._boundary,
          "sync=%s async=%s" % (gr_sync._boundary, gr_async._boundary))

    check("zero-latency async reuses one CUDA event across all checks",
          len(_FakeEvent.created) == 1,
          "created=%d" % len(_FakeEvent.created))

    # Deferred-exact uses the copy but waits at the next decision if needed.
    # Even a never-ready event must therefore give the exact sync decisions,
    # rather than the fail-closed substitutions made by async_check.
    _FakeEvent.created.clear()
    torch.cuda.Event = _FakeEvent
    gr_deferred = _gr(warmup=4, check_every=2, deferred_check=True)
    d_deferred = _run(gr_deferred, STABLE60)
    check("deferred_check preserves the synchronous decision sequence",
          d_sync == d_deferred,
          "diverged at %s" % [i for i in range(len(d_sync))
                               if d_sync[i] != d_deferred[i]][:5])
    check("deferred_check waits on one reusable event, not a stale decision",
          len(_FakeEvent.created) == 1,
          "created=%d" % len(_FakeEvent.created))
    # Safety: an event that never becomes ready must force fresh renders,
    # never reuse from whatever decision preceded the pending check.
    _FakeEvent.created.clear()
    gr_stale = _gr(warmup=4, check_every=2, async_check=True)
    gr_stale.reset_frame()
    stale_decisions = []
    for i, g in enumerate(STABLE60[:10]):
        reuse = gr_stale.should_reuse(i, False)
        stale_decisions.append(reuse)
        if not reuse:
            gr_stale.note_fresh_grad(g, i)
            gr_stale.note_rendered()
        else:
            gr_stale.note_reused()
    check("a never-landing async check fails closed to fresh renders",
          not any(stale_decisions), "decisions=%s" % stale_decisions)
    check("a never-landing check leaves a pending slot, not an exception",
          gr_stale._pending_check is not None,
          "pending=%s" % (gr_stale._pending_check,))
    check("a never-landing check does not advance past its initial boundary",
          gr_stale._boundary == gr_stale.warmup + gr_stale.check_every * gr_stale.period,
          "boundary=%s" % gr_stale._boundary)
    _pending_event = gr_stale._async_event
    check("exactly one event was created for the whole stalled window "
          "(never re-issued while one is in flight)",
          len(_FakeEvent.created) == 1, "created=%d" % len(_FakeEvent.created))

    # Now let it land and confirm the NEXT poll picks it up and applies it.
    #
    # NOT asserting the boundary itself moved: this window closed at
    # iter_idx=1 (captured at ISSUE time, matching the sync path's own
    # semantics exactly - see _poll_pending_check), so its target is
    # 1 + check_every*period = 5, which is already below the frame's
    # initial boundary of warmup + check_every*period = 8. max(8, 5) == 8
    # is the CORRECT answer here, identical to what the synchronous path
    # would also compute from the same iter_idx - a sync run of this exact
    # scenario moves nothing either. What actually distinguishes "applied"
    # from "still pending" is adaptive_extends, not the boundary value.
    _pending_event.ready = True
    extends_before = gr_stale.adaptive_extends
    landing_reuse = gr_stale.should_reuse(10, False)
    check("landing the event clears the pending slot",
          gr_stale._pending_check is None)
    check("landing the event applies the trusted decision (adaptive_extends)",
          gr_stale.adaptive_extends == extends_before + 1,
          "before=%s after=%s" % (extends_before, gr_stale.adaptive_extends))
    check("a landed but backlogged result still fails closed",
          not landing_reuse and not gr_stale._async_trust_ready,
          "reuse=%s ready=%s" %
          (landing_reuse, gr_stale._async_trust_ready))

    # Frame-boundary discard: a pending check from a finished frame must
    # never be allowed to mutate the NEXT frame's freshly-reset state, even
    # if it lands later.
    _FakeEvent.created.clear()
    gr_cross = _gr(warmup=4, check_every=2, async_check=True)
    gr_cross.reset_frame()
    for i, g in enumerate(STABLE60[:10]):
        if not gr_cross.should_reuse(i, False):
            gr_cross.note_fresh_grad(g, i)
            gr_cross.note_rendered()
        else:
            gr_cross.note_reused()
    pending_from_frame1 = gr_cross._pending_check
    check("frame 1 ends with an in-flight (not-yet-landed) check",
          pending_from_frame1 is not None)
    gr_cross.reset_frame()  # frame 2 begins
    check("reset_frame() retains the buffer but invalidates its old token",
          gr_cross._pending_check is not None
          and pending_from_frame1[2] != gr_cross._frame_token)
    boundary_at_frame2_start = gr_cross._boundary
    gr_cross._async_event.ready = True  # it finally "lands", too late
    gr_cross._poll_pending_check()
    check("a stale cross-frame landing does not retroactively touch frame 2",
          gr_cross._pending_check is None
          and gr_cross._boundary == boundary_at_frame2_start,
          "before=%s after=%s" % (boundary_at_frame2_start, gr_cross._boundary))
finally:
    torch.cuda.Event = _orig_event
    torch.empty = _orig_empty


print("all passed" if not _fails else "FAILED: %s" % ", ".join(_fails))
sys.exit(1 if _fails else 0)
