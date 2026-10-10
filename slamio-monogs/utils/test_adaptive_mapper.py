"""AdaptiveMapper: the in-run n_new_pts_ref calibration, and that it changes
nothing when it is off. Pure python, no torch. Run: python utils/test_adaptive_mapper.py
"""
import os
import sys
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.adaptive_mapper import AdaptiveMapper, autoscale_iteration_bounds


def quiet(fn, *a, **k):
    with redirect_stdout(StringIO()) as buf:
        out = fn(*a, **k)
    return out, buf.getvalue()


def make(**over):
    cfg = dict(enabled=True, min_iters=60, max_iters=100, w_unseen=0.5,
               w_new_pts=0.5, w_depth=0.0, w_color=0.0, n_new_pts_ref=5000)
    cfg.update(over)
    am, _ = quiet(AdaptiveMapper, cfg)
    return am


def call(am, raw, ref=None, unseen=0.5):
    ref = am.n_new_pts_ref if ref is None else ref
    ratio = min(raw / max(ref, 1), 1.0)
    return quiet(am.compute_budget, unseen_ratio=unseen, n_new_pts_ratio=ratio,
                 n_new_pts=raw)


print("[0] calibration OFF (default): identical to the old behaviour")
old = make()
new = make(calibrate_new_pts_frames=0)
for raw in (100, 3000, 90000):
    b0, _ = call(old, raw)
    b1, _ = call(new, raw)
    assert b0 == b1, (b0, b1)
assert old.n_new_pts_ref == 5000 and not old._np_fitted
print("    ok")

print("[1] calibration ON: full budget for skip+N frames, then the p90 is installed")
am = make(calibrate_new_pts_frames=50, calibrate_new_pts_skip=5)
counts = [50000] * 5 + list(range(1000, 1000 + 50 * 20, 20))   # 5 warm-up probes, then 50 frames
budgets, out_last = [], ""
for i, raw in enumerate(counts):
    b, out = call(am, raw)
    budgets.append(b)
    if "calibrated n_new_pts_ref" in out:
        out_last = out
assert all(b == 100 for b in budgets), budgets      # whole window at max_iters
assert am._np_fitted and am._np_calibrated
steady = sorted(counts[5:])
expect = steady[int(round(0.9 * (len(steady) - 1)))]
assert am.n_new_pts_ref == expect, (am.n_new_pts_ref, expect)
assert am.n_new_pts_ref < 50000, "the warm-up probes must not be in the fit"
assert "5000 -> %d" % expect in out_last, out_last
print("    ok: ref 5000 ->", am.n_new_pts_ref)

print("[2] after calibration the budget adapts, and the reference stays frozen")
ref_before = am.n_new_pts_ref
b_low, _ = call(am, 10, unseen=0.0)
b_high, _ = call(am, 10 * ref_before, unseen=1.0)
assert b_high == 100 and 60 <= b_low < 100, (b_low, b_high)
call(am, 999999)
assert am.n_new_pts_ref == ref_before
print("    ok: low novelty ->", b_low, ", high ->", b_high)

print("[3] a caller that never supplies the raw count is NOT held at max_iters")
am = make(calibrate_new_pts_frames=50)
bs = [quiet(am.compute_budget, unseen_ratio=0.0, n_new_pts_ratio=0.0)[0] for _ in range(80)]
assert set(bs) == {60}, set(bs)
assert not am._np_fitted
print("    ok: budget", bs[0], "throughout (calibration never started)")

print("[4] a sequence shorter than the window installs nothing and stays at max_iters")
am = make(calibrate_new_pts_frames=50, calibrate_new_pts_skip=5)
bs = [call(am, 2000)[0] for _ in range(30)]
assert all(b == 100 for b in bs) and not am._np_fitted and am.n_new_pts_ref == 5000
print("    ok")

print("[5] the summary reports the raw distribution against the reference in force")
am = make(calibrate_new_pts_frames=10, calibrate_new_pts_skip=2)
for raw in [9000] * 2 + [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800] + [1500] * 20:
    call(am, raw)
s = am.summary()
assert "NEW POINTS over 32 frames" in s and "fitted in-run" in s and "fit/current" in s, s
print("    ok")
print(s)

print("[6] disabled mapper is untouched")
am = make(enabled=False, calibrate_new_pts_frames=50)
assert quiet(am.compute_budget, unseen_ratio=0.0, n_new_pts_ratio=0.0, n_new_pts=5)[0] == 100
assert am._np_raw == []
print("    ok")

print("[7] BOTH calibration windows at once run concurrently, not one after the other")
# Bug: compute_budget() used to return early while the new-points window was
# still open, before ever reaching the depth/colour accumulation code below
# it - so calibration_keyframes couldn't start counting samples until
# calibrate_new_pts_frames had already finished, serialising two windows
# that are supposed to be independent into one roughly twice as long, each
# then fitted over a later, less representative slice than a solo window of
# either kind ever saw. No existing caller before SplaTAM ever set both on
# the same instance (GSLAM: new-points only; MonoGS: depth/colour only), so
# nothing had exercised this path. Deliberately different window sizes here
# (kf finishes at call 13, np at call 35) so "kf finished on schedule, not
# delayed past np's completion" is actually checked, not just "eventually
# both finish".
am = make(w_depth=0.5, w_color=0.0, calibration_keyframes=10, calibration_skip=3,
          calibrate_new_pts_frames=30, calibrate_new_pts_skip=5)
depth_calibrated_at = np_calibrated_at = None
for i in range(40):
    b, out = quiet(am.compute_budget, unseen_ratio=0.5, n_new_pts_ratio=0.5,
                   n_new_pts=1000 + i, depth_error=0.01 + i * 0.0001)
    if depth_calibrated_at is None and am._calibrated:
        depth_calibrated_at = i
    if np_calibrated_at is None and am._np_calibrated:
        np_calibrated_at = i
    if i < 12 or i < 34:
        assert b == 100, (i, b)   # neither window complete yet -> full budget
assert depth_calibrated_at == 12, depth_calibrated_at   # 3 skip + 10 kf, 0-indexed
assert np_calibrated_at == 34, np_calibrated_at          # 5 skip + 30 np, 0-indexed
assert len(am._depth_errs) == 40, "depth accumulation must not have been blocked by the np window"
print("    ok: depth_error_ref calibrated at call", depth_calibrated_at,
      "(not delayed to after np's call", np_calibrated_at, ")")

print("[8] MonoGS bounds follow the detected async/inline mapping ceiling")
cfg = dict(enabled=True, min_iters=3, max_iters=10)
async_cfg, async_info = autoscale_iteration_bounds(cfg, 10)
inline_cfg, inline_info = autoscale_iteration_bounds(cfg, 150)
assert (async_cfg["min_iters"], async_cfg["max_iters"]) == (3, 10)
assert not async_info["scaled"]
assert (inline_cfg["min_iters"], inline_cfg["max_iters"]) == (45, 150)
assert inline_info["scaled"]
assert cfg == dict(enabled=True, min_iters=3, max_iters=10), "input must not mutate"

absolute, absolute_info = autoscale_iteration_bounds(
    dict(enabled=True, min_iters=3, max_iters=10, auto_scale_bounds=False), 150)
assert (absolute["min_iters"], absolute["max_iters"]) == (3, 10)
assert not absolute_info["scaled"]
print("    ok: async 3..10; inline 45..150; explicit opt-out stays 3..10")
print("ALL PASS")
