"""tangent_of_pose_delta_batch must equal tangent_of_pose_delta row by row.

    python utils/test_tangent_batch.py

The batched version exists only to remove per-row Python dispatch from the
stopping bookkeeping. It is a pure performance change, so the bar is numerical
equivalence with the single-row function across the motion band tracking
actually produces, plus the two properties the single-row docstring calls
load-bearing: float64 promotion of float32 inputs, and invariance to an
unnormalised quaternion scale.
"""
import math
import sys
import time

import torch

sys.path.insert(0, ".")
from utils.pose_preconditioner import (  # noqa: E402
    tangent_of_pose_delta,
    tangent_of_pose_delta_batch,
)

torch.manual_seed(0)
FAIL = []


def check(name, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def rand_quat(n):
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def small_rot_quat(n, angle):
    axis = torch.randn(n, 3, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half = torch.full((n, 1), angle / 2.0, dtype=torch.float64)
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def per_row(q0, t0, q1, t1):
    return torch.stack([tangent_of_pose_delta(q0[i], t0[i], q1[i], t1[i])
                        for i in range(q0.shape[0])])


N = 256
print("equivalence against the single-row function")
for angle in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.3, 1.0):
    q0 = rand_quat(N)
    t0 = torch.randn(N, 3, dtype=torch.float64)
    q1 = quat_mul(small_rot_quat(N, angle), q0)
    t1 = t0 + torch.randn(N, 3, dtype=torch.float64) * angle
    ref = per_row(q0, t0, q1, t1)
    got = tangent_of_pose_delta_batch(q0, t0, q1, t1)
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    check("angle %-6g" % angle, err <= 1e-12 * max(scale, 1e-3),
          "max |batch - row| = %.2e (|dxi| max %.2e)" % (err, scale))

print("float32 inputs are promoted, as in the single-row function")
q0 = rand_quat(N).float()
t0 = torch.randn(N, 3).float()
q1 = quat_mul(small_rot_quat(N, 1e-4), q0.double()).float()
t1 = (t0 + torch.randn(N, 3) * 1e-4).float()
ref = per_row(q0, t0, q1, t1)
got = tangent_of_pose_delta_batch(q0, t0, q1, t1)
check("dtype float64", got.dtype == torch.float64, str(got.dtype))
err = (got - ref).abs().max().item()
check("float32-in equivalence", err <= 1e-12, "max diff %.2e" % err)

print("unnormalised quaternion scale is a gauge and must not register")
q0 = rand_quat(N)
t0 = torch.randn(N, 3, dtype=torch.float64)
d = tangent_of_pose_delta_batch(q0, t0, q0 * 1.7, t0).norm(dim=-1).max().item()
check("q * 1.7 reads as zero motion", d < 1e-12, "max |dxi| = %.1e" % d)

print("shape")
got = tangent_of_pose_delta_batch(rand_quat(8), torch.zeros(8, 3, dtype=torch.float64),
                                  rand_quat(8), torch.zeros(8, 3, dtype=torch.float64))
check("(8,4),(8,3) -> (8,6)", tuple(got.shape) == (8, 6), str(tuple(got.shape)))
check("finite", bool(torch.isfinite(got).all()))

print("timing on CPU, one drained batch of 8 rows, single thread")
torch.set_num_threads(1)
q0, t0 = rand_quat(8).float(), torch.randn(8, 3).float()
q1 = quat_mul(small_rot_quat(8, 1e-4), q0.double()).float()
t1 = (t0 + torch.randn(8, 3) * 1e-4).float()
reps = 400
for _ in range(20):
    per_row(q0, t0, q1, t1)
    tangent_of_pose_delta_batch(q0, t0, q1, t1)
t = time.perf_counter()
for _ in range(reps):
    per_row(q0, t0, q1, t1).detach().cpu().numpy()
row_ms = (time.perf_counter() - t) / (reps * 8) * 1e3
t = time.perf_counter()
for _ in range(reps):
    tangent_of_pose_delta_batch(q0, t0, q1, t1).detach().cpu().numpy()
bat_ms = (time.perf_counter() - t) / (reps * 8) * 1e3
print("  per-row loop  %.4f ms/row" % row_ms)
print("  batched       %.4f ms/row  (%.1fx)" % (bat_ms, row_ms / bat_ms))
print("  at 137 it/frame: %.0f -> %.0f ms/frame" % (row_ms * 137, bat_ms * 137))

print()
if FAIL:
    print("FAILED: %s" % ", ".join(FAIL))
    sys.exit(1)
print("ALL PASS")
