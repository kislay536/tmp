"""Focused CPU tests for the opt-in first-excursion diagnostic."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_preconditioner import PosePreconditioner
from utils.preconditioner_excursion import FirstExcursionRecorder


def check(ok, message):
    if not ok:
        raise AssertionError(message)
    print(f"PASS  {message}")


with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "trace.jsonl")
    recorder = FirstExcursionRecorder(path, history=4, post=2,
                                      raw_step_ratio=4.0,
                                      seen_drop_ratio=0.0)
    for iteration, ratio in enumerate((0.2, 0.3, 5.0, 0.4, 0.3)):
        recorder.record({
            "frame": 7,
            "iteration": iteration,
            "loss": 1.0,
            "seen": 100,
            "preconditioner": {"raw_step_ratio": ratio, "live": 1.0},
        })
    check(recorder.written, "trigger writes after the requested post tail")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    check(rows[0]["trigger"] == {
        "frame": 7, "iteration": 2, "reasons": ["raw_step_ratio"]},
        "metadata identifies the first excursion")
    check([row["iteration"] for row in rows[1:]] == [1, 2, 3, 4],
          "bounded history keeps pre-trigger context and the post tail")

    quiet = FirstExcursionRecorder(os.path.join(tmp, "quiet.jsonl"),
                                   raw_step_ratio=4.0,
                                   seen_drop_ratio=0.0)
    quiet.record({"frame": 0, "iteration": 0,
                  "preconditioner": {"raw_step_ratio": 0.2, "live": 1.0}})
    quiet.close()
    check(not quiet.written and not os.path.exists(quiet.path),
          "a healthy run writes no misleading excursion file")

    scheduled_path = os.path.join(tmp, "scheduled.jsonl")
    scheduled = FirstExcursionRecorder(
        scheduled_path, history=8, post=1, raw_step_ratio=0.0,
        seen_drop_ratio=0.0, trigger_frame=4, trigger_iteration=2)
    for frame, iteration in ((3, 9), (4, 0), (4, 1), (4, 2), (4, 3)):
        scheduled.record({
            "frame": frame,
            "iteration": iteration,
            "preconditioner": {"raw_step_ratio": 0.1, "live": 1.0},
        })
    scheduled_rows = [json.loads(line) for line in
                      open(scheduled_path, encoding="utf-8")]
    check(scheduled_rows[0]["trigger"] == {
        "frame": 4, "iteration": 2, "reasons": ["scheduled"]},
        "a scheduled frame and iteration force a diagnostic trace")
    check(scheduled_rows[0]["trigger_frame"] == 4
          and scheduled_rows[0]["trigger_iteration"] == 2,
          "scheduled trigger settings are preserved in metadata")

    partial = FirstExcursionRecorder(os.path.join(tmp, "partial.jsonl"),
                                     post=20, raw_step_ratio=0.0,
                                     seen_drop_ratio=0.0)
    partial.record({"frame": 3, "iteration": 9,
                    "preconditioner": {"raw_step_ratio": 0.0, "live": 0.0}})
    partial.close()
    check(partial.written, "close flushes a trigger near the end of a run")


kw = dict(dim=6, lr=0.004, refactor_every=1, carry=False,
          dtype=torch.float64, device="cpu")
plain = PosePreconditioner(**kw)
traced = PosePreconditioner(**kw, trace_steps=True)
for i in range(8):
    grad = torch.tensor([1.0 + i, -0.5, 0.25, 0.75, -1.25, 0.4],
                        dtype=torch.float64)
    d0 = plain.step(grad)
    d1 = traced.step(grad)
    check(torch.equal(d0, d1),
          f"trace buffers do not change step arithmetic at iteration {i}")
    plain.refactor()
    traced.refactor()

check(torch.equal(plain.m, traced.m) and torch.equal(plain.M, traced.M),
      "trace buffers leave optimiser state bit-identical")
snapshot = traced.diagnostic_snapshot()
check(len(snapshot["eigenvalues"]) == 6
      and len(snapshot["spectral_step_coefficients"]) == 6,
      "snapshot exposes the six spectral directions")
check(len(snapshot["M"]) == 6 and len(snapshot["M"][0]) == 6,
      "snapshot exposes the exact 6x6 metric")
check(snapshot["raw_step_norm"] >= snapshot["applied_step_norm"],
      "snapshot distinguishes requested and applied step magnitudes")

print("all excursion-recorder checks passed")
