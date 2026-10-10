"""The bypass contract: a bypassed iteration must not touch graph state.

WHY THIS EXISTS. Gradient reuse makes the tracking iteration body branch, and
a CUDA graph records ONE fixed kernel sequence - so the two were refused
together. `bypass=True` is what lets them compose: the reuse iterations run
eager, the rendering ones replay, and no second capture is paid for.

The whole correctness argument is an ORDERING one - bypass returns before the
phase check, before the replay, and before the warmup counter - and an ordering
argument is exactly the kind that survives a refactor silently broken. If the
early return ever moves below the warmup counter, reuse iterations start
consuming warmup slots, the capture lands on whatever iteration happens to be
next, and a graph recorded on a rendering iteration can be replayed on a
reusing one: the original failure mode, back, and invisible except as a
speedup that is not there.

Runs on CPU. Everything tested here happens before any CUDA call, which is
precisely the property being asserted.

    python utils/test_tracking_iteration_graph.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from utils.tracking_iteration_graph import (  # noqa: E402
    TrackingIterationGraph, _CaptureNoGC, _gc_collect_suppressed, _gc_pause,
    _gc_resume)


def _graph(**cfg):
    cfg.setdefault("enabled", True)
    cfg.setdefault("warmup_iters", 3)
    return TrackingIterationGraph(cfg)


def test_bypass_runs_eagerly_and_leaves_state_alone():
    g = _graph()
    calls = []
    for i in range(10):
        out = g.run(lambda: calls.append(i) or ("out", i), bypass=True)
        assert out == ("out", i), out
    assert calls == list(range(10)), calls
    # The three pieces of state the ordering argument protects.
    assert g._warmup_done == 0, g._warmup_done
    assert g._graph is None
    assert g._outputs is None
    assert g.replays == 0, g.replays
    assert g.bypassed == 10, g.bypassed
    print("  PASS  bypass runs eagerly and advances no graph state")


def test_bypass_ignores_phase():
    """A phase change on a bypassed iteration is DEFERRED, not lost.

    The graph only has to change when it is about to be used, so a switch seen
    on an iteration that never touches the graph should not burn a capture.
    The next non-bypassed iteration sees the new phase and recaptures then.
    """
    g = _graph()
    for i in range(10):
        g.run(lambda: None, phase=i % 2, bypass=True)
    assert g.phase_switches == 0, g.phase_switches
    assert g._phase == 0, g._phase
    # ... and the deferred switch still lands once a real iteration runs.
    g.run(lambda: None, phase=1)
    assert g.phase_switches == 1, g.phase_switches
    print("  PASS  a phase change on a bypassed iteration is deferred, not lost")


def test_warmup_counts_only_non_bypassed():
    """The capture lands on a rendering iteration BY CONSTRUCTION.

    This is the property that makes a second graph unnecessary. Interleave
    bypassed and real iterations and the warmup counter must advance only on
    the real ones.
    """
    # warmup_iters stays ABOVE the number of real iterations here on purpose:
    # a capture needs a CUDA context, and the point of this file is that none
    # of the bypass logic does. Crossing the warmup boundary would make the
    # test require a GPU to assert something that happens on the host.
    g = _graph(warmup_iters=8)
    for i in range(12):
        g.run(lambda: None, bypass=(i % 2 == 1))
    assert g.bypassed == 6, g.bypassed
    # Twelve iterations, six of them real - and only those six count.
    assert g._warmup_done == 6, g._warmup_done
    print("  PASS  warmup consumes only non-bypassed iterations")


def test_disabled_ignores_bypass():
    g = _graph(enabled=False)
    assert g.run(lambda: "x", bypass=True) == "x"
    assert g.run(lambda: "x", bypass=False) == "x"
    assert g.bypassed == 0, g.bypassed
    print("  PASS  bypass is inert when the graph is disabled")


def test_summary_reports_the_denominator():
    """replays/frame is not replays/iteration once anything is bypassed."""
    g = _graph()
    for _ in range(4):
        g.run(lambda: None, bypass=True)
    s = g.summary()
    assert "4 iterations bypassed" in s, s
    assert "graph-eligible" in s, s
    print("  PASS  summary reports bypassed iterations and the replay base")


def test_will_capture_predicts_the_capture_iteration():
    """Callers use this to drop live references BEFORE the capture runs.

    A tensor still referenced from iteration N-1 keeps its warmup-era autograd
    accumulators alive, and the capture then reuses nodes stamped with the
    legacy default stream - which fails inside backward(), nowhere near the
    line holding the reference. Gradient reuse holds exactly such a reference
    by design, so the predicate has to be true on the capture iteration and
    false on every other one.
    """
    g = _graph(warmup_iters=3)
    seen = []
    for _ in range(3):
        seen.append(g.will_capture())
        g.run(lambda: None)
    # Warmup iterations: nothing to capture yet.
    assert seen == [False, False, False], seen
    # Warmup is now exhausted, so the NEXT real iteration captures.
    assert g.will_capture() is True
    # A bypassed iteration does not consume it - the capture is still pending.
    g.run(lambda: None, bypass=True)
    assert g.will_capture() is True
    print("  PASS  will_capture marks the capture iteration, and bypass "
          "does not consume it")


def test_gc_suppression_skips_the_call_and_restores_it():
    """The capture's gc.collect() is skipped, and exactly that call.

    Automatic collection is disabled for the duration of the check only so a
    threshold-triggered pass cannot free the cycle behind the test's back.
    """
    import gc
    import weakref

    class Node:
        pass

    def make_cycle():
        a, b = Node(), Node()
        a.other, b.other = b, a
        return weakref.ref(a)

    real = gc.collect
    gc.disable()
    try:
        ref = make_cycle()
        with _gc_collect_suppressed():
            assert gc.collect() == 0
            assert ref() is not None, "the cycle was collected despite the skip"
        assert gc.collect is real, "gc.collect was not restored"
        gc.collect()
        assert ref() is None, "collection does not work again after the block"
        try:
            with _gc_collect_suppressed():
                raise ValueError("boom")
        except ValueError:
            pass
        assert gc.collect is real, "an exception left gc.collect patched"
    finally:
        gc.enable()
    print("  PASS  gc suppression skips the explicit call and always restores it")


def test_skip_gc_is_off_by_default_and_env_overrides():
    saved = os.environ.pop("ITERGRAPH_SKIP_GC", None)
    try:
        assert TrackingIterationGraph({"enabled": False}).skip_gc is False
        assert TrackingIterationGraph(
            {"enabled": False, "skip_gc": True}).skip_gc is True
        os.environ["ITERGRAPH_SKIP_GC"] = "1"
        assert TrackingIterationGraph({"enabled": False}).skip_gc is True
        # The environment wins over the config, in both directions.
        os.environ["ITERGRAPH_SKIP_GC"] = "0"
        assert TrackingIterationGraph(
            {"enabled": False, "skip_gc": True}).skip_gc is False
        os.environ["ITERGRAPH_SKIP_GC"] = "1"
        assert "gc.collect" in _graph(skip_gc=True).summary()
    finally:
        os.environ.pop("ITERGRAPH_SKIP_GC", None)
        if saved is not None:
            os.environ["ITERGRAPH_SKIP_GC"] = saved
    assert issubclass(_CaptureNoGC, torch.cuda.graph)
    print("  PASS  skip_gc is off by default, the environment overrides the "
          "config, and the summary says when it is on")


def test_gc_pause_stops_automatic_collection_and_restores_it():
    """The collector must be OFF for the whole capture window, then back.

    Skipping only the explicit gc.collect() let a threshold-triggered pass
    run inside a capture and abort MonoGS mid-run. The pause has to hold
    across the window and restore the PRIOR state - including 'was already
    off', which must not be switched on behind the caller's back.
    """
    import gc

    gc.enable()
    was = _gc_pause()
    try:
        assert was is True
        assert not gc.isenabled(), 'automatic gc still on inside the window'
    finally:
        _gc_resume(was)
    assert gc.isenabled(), 'gc was not switched back on'

    gc.disable()
    try:
        was = _gc_pause()
        assert was is False
        _gc_resume(was)
        assert not gc.isenabled(), 'resume enabled a collector that was off'
    finally:
        gc.enable()
    print("  PASS  automatic gc is paused for the capture window and the "
          "prior state is restored exactly")


def test_capture_context_pauses_gc_across_enter_and_exit():
    """The capture context must pause on enter and resume on exit AND on error.

    A real capture needs CUDA, so this pins the wiring by name: dropping the
    pause, or the resume from __exit__, silently brings back the
    gc-inside-capture abort."""
    enter_names = _CaptureNoGC.__enter__.__code__.co_names
    exit_names = _CaptureNoGC.__exit__.__code__.co_names
    assert '_gc_pause' in enter_names, enter_names
    assert '_gc_resume' in enter_names, 'a failing __enter__ must resume gc'
    assert '_gc_resume' in exit_names, exit_names
    print("  PASS  the capture context pauses gc on enter and resumes it on "
          "exit and on a failed enter")


if __name__ == "__main__":
    print("tracking iteration graph: bypass contract")
    test_bypass_runs_eagerly_and_leaves_state_alone()
    test_bypass_ignores_phase()
    test_warmup_counts_only_non_bypassed()
    test_will_capture_predicts_the_capture_iteration()
    test_disabled_ignores_bypass()
    test_summary_reports_the_denominator()
    test_gc_suppression_skips_the_call_and_restores_it()
    test_skip_gc_is_off_by_default_and_env_overrides()
    test_gc_pause_stops_automatic_collection_and_restores_it()
    test_capture_context_pauses_gc_across_enter_and_exit()
    print("all passed")
