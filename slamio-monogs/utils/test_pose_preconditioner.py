"""
test_pose_preconditioner.py - validate the maths before it touches a tracker.

Run:  python utils/test_pose_preconditioner.py

CPU only, no GPU, no SLAM. Every test has an answer known in closed form or by
construction, in the style of the gn_tracking validation (planted minima
recovered exactly, generators against central differences, cos = 1.0000000000
on an exactly-differentiable case). The GN campaign's most expensive mistakes
were all things a test like this would have caught in seconds - a double
transform in a render, an assert that fired on exactly the frames that needed
explaining, a series cutoff that was never taken when it mattered.

TEST 4 IS THE ONE THAT MATTERS. Rotation covariance is the property that
separates a full matrix from a diagonal one, and it is the entire claim of the
method: a diagonal preconditioner is tied to the coordinate axes it was written
in, so it cannot represent coupling between pose DOF no matter how it is tuned.
If test 4 fails, the implementation is not doing the thing the thesis says it
does, whatever the SLAM numbers come out as.
"""

from __future__ import annotations

import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_preconditioner import (  # noqa: E402
    PosePreconditioner, _inv_sqrt_damped, se3_adjoint, transport_metric,
    _quat_to_mat, quat_trans_grad_to_tangent, per_gaussian_tangent_grads,
    sample_metric, effective_samples, rank1_alignment,
    bfgs_update, clamp_spectrum,
    se3_exp_capturable, se3_log_capturable, tangent_of_pose_delta,
    apply_tangent_step, mat_to_quat_capturable,
)
from utils.gn_tracking import se3_exp, se3_log  # noqa: E402

torch.manual_seed(0)
DT = torch.float64
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
print("\n[1] damped inverse square root against the closed form")
# Plant a spectrum, so the expected answer is written down rather than
# recomputed by the same code path under test.
evals = torch.tensor([1e-3, 1e-2, 5e-2, 7e-2, 7e-1, 1.0], dtype=DT)
Q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=DT))
M = (Q * evals) @ Q.T
tau, eps = 0.05, 1e-12

P = _inv_sqrt_damped(M, tau, eps)
lmax = evals.max()
want_inv = 1.0 / ((evals + tau * lmax).sqrt() + eps)
P_want = (Q * want_inv) @ Q.T
check("matches the Levenberg-floored inverse square root",
      torch.allclose(P, P_want, atol=1e-8),
      f"max|dP| = {(P - P_want).abs().max():.3e}")

# Low curvature SHOULD buy a longer step - that is preconditioning. What must
# never happen is an UNBOUNDED one: a shrinkage-style damping sends an
# unobserved direction's scaling to 1/eps, and that is the 1e20 divergence this
# file caught on its first run. With the Levenberg floor the ratio between the
# longest and shortest step is exactly sqrt((1 + tau)/tau), independent of how
# singular M is.
P_worst = _inv_sqrt_damped(torch.outer(torch.randn(6, dtype=DT),
                                       torch.zeros(6, dtype=DT)).add_(
                               torch.diag(torch.tensor([1., 0, 0, 0, 0, 0], dtype=DT))),
                           tau, eps)
ev = torch.linalg.eigvalsh(P_worst)
check("step-length ratio is bounded by sqrt((1+tau)/tau) even on a rank-1 M",
      abs(float(ev.max() / ev.min()) - math.sqrt((1 + tau) / tau)) < 1e-4,
      f"ratio = {float(ev.max() / ev.min()):.4f} vs {math.sqrt((1 + tau) / tau):.4f}")

# A singular M is the NORMAL case, not an edge case: gg^T is rank 1, so after
# k updates from cold M has rank <= k and is singular until k >= n.
M_rank1 = torch.outer(torch.randn(6, dtype=DT), torch.ones(6, dtype=DT))
M_rank1 = M_rank1 @ M_rank1.T
P_sing = _inv_sqrt_damped(M_rank1, tau, eps)
check("finite on a rank-1 (singular) M",
      bool(torch.isfinite(P_sing).all()),
      f"cond = {torch.linalg.cond(P_sing):.3e}")

P_zero = _inv_sqrt_damped(torch.zeros(6, 6, dtype=DT), tau, eps)
check("identity on a zero M (cold start)",
      torch.allclose(P_zero, torch.eye(6, dtype=DT), atol=1e-12))


# ---------------------------------------------------------------------------
print("\n[2] SE(3) adjoint against its defining property")
# Ad_T is defined by  exp(Ad_T xi) = T exp(xi) T^-1.  Checking against that,
# rather than against the [[R, [t]xR],[0,R]] formula, means a transposed block
# or a swapped convention cannot pass.
xi = torch.tensor([0.02, -0.01, 0.03, 0.004, -0.002, 0.005], dtype=DT)
T = se3_exp(torch.tensor([0.3, -0.2, 0.5, 0.1, 0.2, -0.15], dtype=DT))
A = se3_adjoint(T)
lhs = A @ xi
rhs = se3_log(T @ se3_exp(xi) @ torch.linalg.inv(T))
check("exp(Ad_T xi) == T exp(xi) T^-1",
      torch.allclose(lhs, rhs, atol=1e-9),
      f"max|d| = {(lhs - rhs).abs().max():.3e}")

# Under pure rotation the adjoint is block-diagonal and orthogonal; under
# translation it is not. That asymmetry is exactly why transport is a real
# correction between consecutive frames rather than a relabelling.
T_rot = se3_exp(torch.tensor([0., 0., 0., 0.1, 0.2, -0.15], dtype=DT))
A_rot = se3_adjoint(T_rot)
check("orthogonal under pure rotation",
      torch.allclose(A_rot @ A_rot.T, torch.eye(6, dtype=DT), atol=1e-12))
check("NOT orthogonal once translation is present",
      not torch.allclose(A @ A.T, torch.eye(6, dtype=DT), atol=1e-6),
      f"max|AA^T - I| = {(A @ A.T - torch.eye(6, dtype=DT)).abs().max():.3e}")


# ---------------------------------------------------------------------------
print("\n[3] metric transport")
M6 = (Q * evals) @ Q.T
M_fwd = transport_metric(M6, A)
M_back = transport_metric(M_fwd, torch.linalg.inv(A))
check("round trip A then A^-1 is identity",
      torch.allclose(M_back, M6, atol=1e-10),
      f"max|dM| = {(M_back - M6).abs().max():.3e}")
check("matches A^-T M A^-1 directly",
      torch.allclose(M_fwd, torch.linalg.inv(A).T @ M6 @ torch.linalg.inv(A), atol=1e-10))
check("stays symmetric",
      torch.allclose(M_fwd, M_fwd.T, atol=1e-12))

# Transport must act on gradients the way it claims: if g transforms as a
# covector, the transported M must equal the second moment of transported g.
g_prev = torch.randn(6, dtype=DT)
g_new = torch.linalg.inv(A).T @ g_prev
check("consistent with g_t = A^-T g_{t-1}",
      torch.allclose(transport_metric(torch.outer(g_prev, g_prev), A),
                     torch.outer(g_new, g_new), atol=1e-10))


# ---------------------------------------------------------------------------
print("\n[4] ROTATION COVARIANCE - the property a diagonal method cannot have")
# Ill-conditioned quadratic, and the same problem stated in a rotated basis.
# A coordinate-free optimiser must produce the same trajectory up to that
# rotation. Adam cannot: its second moment is tied to the axes.
H = (Q * torch.tensor([1e-3, 1e-2, 5e-2, 7e-2, 7e-1, 1.0], dtype=DT)) @ Q.T
R, _ = torch.linalg.qr(torch.randn(6, 6, dtype=DT))


def run(precond, rotate, n_steps=40):
    """Trajectory in the ORIGINAL basis, optimising in the (possibly) rotated one."""
    Hl = R.T @ H @ R if rotate else H
    x = R.T @ torch.ones(6, dtype=DT) if rotate else torch.ones(6, dtype=DT)
    out = []
    for _ in range(n_steps):
        g = Hl @ x
        if precond is not None:
            x = x + precond.step(g)
            precond.refactor(force=True)
        else:                                   # diagonal Adam, for contrast
            x = x - 0.05 * g / (g.abs() + 1e-8)
        out.append(R @ x if rotate else x.clone())
    return torch.stack(out)


kw = dict(dim=6, lr=0.05, tau=0.05, refactor_every=1, carry=False, dtype=DT)
plain = run(PosePreconditioner(**kw), rotate=False)
rot = run(PosePreconditioner(**kw), rotate=True)
check("full-matrix trajectory is basis independent",
      torch.allclose(plain, rot, atol=1e-8),
      f"max|dx| = {(plain - rot).abs().max():.3e}")

d_plain, d_rot = run(None, rotate=False), run(None, rotate=True)
check("diagonal method is NOT (the contrast this is measured against)",
      not torch.allclose(d_plain, d_rot, atol=1e-6),
      f"max|dx| = {(d_plain - d_rot).abs().max():.3e}")


# ---------------------------------------------------------------------------
print("\n[5] convergence on the ill-conditioned quadratic")
# 1000x spread, matching the measured pose spectrum's shape (lambda/lambda_max
# from 0.010 to 1.000, a real gap after rank 3).
def solve_err(precond, n_steps, lr):
    x = torch.ones(6, dtype=DT)
    for _ in range(n_steps):
        g = H @ x
        if precond is not None:
            x = x + precond.step(g)
            precond.refactor(force=True)
        else:
            x = x - lr * g
    return float(x.norm())


# EACH METHOD AT A SENSIBLE STEP SIZE. GD runs near ITS stability limit, so
# the preconditioner must too or the comparison measures step size rather than
# the metric. kw's lr=0.05 is far below the optimum here and was chosen for the
# basis-independence tests above, where lr cannot affect the assertion.
_kw5 = dict(kw, lr=0.5)
err_pre = solve_err(PosePreconditioner(**_kw5), 60, None)
err_gd = solve_err(None, 60, 0.5)          # near the stability limit for lambda_max = 1
check("beats gradient descent at equal step count",
      err_pre < err_gd,
      f"|x|: preconditioned {err_pre:.3e} vs GD {err_gd:.3e}")

# THE TRAP THIS TEST FELL INTO, PINNED SO IT CANNOT BE RE-FUDGED.
#
# A COLD EMA UNDER-ESTIMATES ITS OWN SCALE, AND THAT ACTS AS A HIDDEN lr
# MULTIPLIER. M is (1-beta2^t) times its true size early, so P ~ M^-1/2 is
# 1/sqrt(1-beta2^t) too large - 4.5x at t=1, tau=0.05. At an UNDER-TUNED lr
# that inflation flatters the method: this check ran at lr=0.05 and the
# uncorrected version "won" (1.41 vs 1.61) purely by taking bigger steps. At a
# sensible lr the corrected one wins (0.54 vs 0.62).
#
# It stayed invisible in the shipped configuration because carry=True means M
# is only cold on frame 1. With carry=False it is cold EVERY frame, which is
# how it was found: step |d| = 1.51x Adam, pinned to the trust region, the pose
# out of the frustum and renders coming back empty.
def _cold(flag, lr):
    pre = PosePreconditioner(**dict(kw, lr=lr))
    pre._M_cold = flag
    return solve_err(pre, 60, None)


check("  a cold uncorrected M inflates the step - the hidden lr multiplier",
      _cold(False, 0.05) < _cold(True, 0.05),
      f"uncorrected {_cold(False, 0.05):.3e} < corrected {_cold(True, 0.05):.3e} "
      f"at an UNDER-TUNED lr")
check("  and the correction is what wins once lr is sensible",
      _cold(True, 0.5) < _cold(False, 0.5),
      f"corrected {_cold(True, 0.5):.3e} < uncorrected {_cold(False, 0.5):.3e}")


# ---------------------------------------------------------------------------
print("\n[6] the carry, and why it is the point")
# Cold: rank(M) <= number of steps taken, so M is singular for the first n-1.
cold = PosePreconditioner(**kw)
for i in range(3):
    cold.step(torch.randn(6, dtype=DT))
rank_cold = int(torch.linalg.matrix_rank(cold.M, atol=1e-14))
check("cold M is rank deficient after 3 steps", rank_cold <= 3, f"rank = {rank_cold}")

# Warm: the carry hands the next frame a full-rank metric on step 1, which is
# the objection to full-matrix methods at short horizons, removed.
warm = PosePreconditioner(**{**kw, "carry": True})
for i in range(12):
    warm.step(torch.randn(6, dtype=DT))
warm.carry_frame(transport=A)
rank_warm = int(torch.linalg.matrix_rank(warm.M, atol=1e-14))
check("carried M is full rank at the start of the next frame",
      rank_warm == 6, f"rank = {rank_warm}")
check("carry resets the first moment, keeps the metric",
      bool(warm.m.abs().max() == 0) and bool(warm.M.abs().max() > 0))

# A transport changes M's coordinate basis before the next frame's first
# update. The cached inverse square root must change at the same time; waiting
# for refactor() would apply one step with P_old and M_new.
_M_before = warm.M.clone()
_refactors_before = warm._refactors
_A_boundary = se3_adjoint(se3_exp(torch.tensor(
    [0.4, -0.2, 0.1, 0.15, -0.08, 0.12], dtype=DT)))
warm.carry_frame(transport=_A_boundary)
_M_expected = transport_metric(_M_before, _A_boundary)
_P_expected = _inv_sqrt_damped(
    _M_expected, warm.tau, warm.eps, warm.shrink)
check("transport refreshes the cached factor before the first new-frame step",
      torch.allclose(warm.M, _M_expected, atol=1e-12)
      and torch.allclose(warm.P, _P_expected, atol=1e-12)
      and warm._refactors == _refactors_before + 1,
      "P_old must never be applied to a gradient expressed beside M_new")

# Gaussian-SLAM stores W_t = W_{t-1} L_t and perturbs L_t on the left. Its
# consecutive local tangents are therefore related by Ad_{L_t^-1}. Pin this
# from the defining equality of the two induced absolute perturbations rather
# than from the block formula for the adjoint.
_W_ref = se3_exp(torch.tensor(
    [0.3, -0.1, 0.2, 0.05, 0.12, -0.09], dtype=DT))
_L_prev = se3_exp(torch.tensor(
    [0.02, 0.01, -0.03, -0.01, 0.02, 0.015], dtype=DT))
_W_cur = _W_ref @ _L_prev
_xi_prev = torch.tensor(
    [0.01, -0.02, 0.005, 0.003, -0.004, 0.002], dtype=DT)
_xi_next = se3_adjoint(torch.linalg.inv(_L_prev)) @ _xi_prev
_old_absolute_action = _W_ref @ se3_exp(_xi_prev) @ torch.linalg.inv(_W_ref)
_new_absolute_action = _W_cur @ se3_exp(_xi_next) @ torch.linalg.inv(_W_cur)
check("Gaussian-SLAM relative tangents transport with Ad_{L_t^-1}",
      torch.allclose(_old_absolute_action, _new_absolute_action, atol=1e-11),
      f"max|dT| = {(_old_absolute_action - _new_absolute_action).abs().max():.3e}")

no_carry = PosePreconditioner(**{**kw, "carry": False})
for i in range(12):
    no_carry.step(torch.randn(6, dtype=DT))
no_carry.carry_frame()
check("carry=False resets the metric instead",
      bool(no_carry.M.abs().max() == 0))


# ---------------------------------------------------------------------------
print("\n[7] scale invariance (the property that makes a stale carry harmless)")
# g -> c g gives m -> c m, M -> c^2 M, P -> c^-1 P, so the step is unchanged.
# This is why carrying M across frames with different gradient magnitudes is
# safe: only its SHAPE transfers, and the magnitude cancels.
a = PosePreconditioner(**kw)
b = PosePreconditioner(**kw)
gs = [torch.randn(6, dtype=DT) for _ in range(15)]
for g in gs:
    da = a.step(g); a.refactor(force=True)
    db = b.step(g * 1e4); b.refactor(force=True)
check("step is invariant to gradient scale",
      torch.allclose(da, db, rtol=1e-6, atol=1e-12),
      f"max|d| = {(da - db).abs().max():.3e}")


# ---------------------------------------------------------------------------
print("\n[9] STEP MAGNITUDE IS BOUNDED - the failure that reached the VM")
# What happened: P is the identity until the first refactor, so the opening
# steps were `-lr * I @ m` - raw, unscaled gradient descent. On a loss summed
# over ~300k pixels that measured 4258x Adam's step norm and threw the
# quaternion to 2915 by frame 5. Adam cannot do this: it divides by sqrt(v), so
# its step is bounded at ~lr per coordinate whatever the gradient magnitude is.
#
# The bound this asserts is the one Adam gets for free and a preconditioner
# does not: step norm stays within a fixed multiple of lr*sqrt(n) for ANY
# gradient scale.
LR, N = 0.002, 6
adam_ref = LR * math.sqrt(N)

for gscale in (1e-8, 1.0, 1e3, 1e8):
    p = PosePreconditioner(dim=N, lr=LR, tau=0.05, refactor_every=10,
                           max_step_mult=10.0, carry=False, dtype=DT)
    worst = 0.0
    for i in range(40):
        d = p.step(torch.randn(N, dtype=DT) * gscale)
        p.refactor()
        worst = max(worst, float(d.norm()))
    check(f"|g| ~ {gscale:.0e}: step stays within 10x Adam",
          worst <= 10.0 * adam_ref * 1.001,
          f"worst {worst:.3e} vs cap {10.0 * adam_ref:.3e}")

# The first step in particular - the one that used to be unscaled.
p = PosePreconditioner(dim=N, lr=LR, tau=0.05, carry=False, dtype=DT)
d0 = p.step(torch.randn(N, dtype=DT) * 1e6)
check("first step (no metric yet) is normalised, not raw GD",
      abs(float(d0.norm()) - adam_ref) < 1e-12,
      f"|d0| = {float(d0.norm()):.3e}, Adam ref {adam_ref:.3e}")

# And it must still point along the gradient - bounded, not mangled.
g0 = torch.randn(N, dtype=DT)
p2 = PosePreconditioner(dim=N, lr=LR, carry=False, dtype=DT)
d1 = p2.step(g0)
cos = float(torch.dot(-d1, g0) / (d1.norm() * g0.norm()))
check("first step still points downhill", abs(cos - 1.0) < 1e-12, f"cos = {cos:.10f}")

# A metric exists from step 1 onward, so no iteration runs unpreconditioned.
p3 = PosePreconditioner(dim=N, lr=LR, refactor_every=10, carry=False, dtype=DT)
p3.step(torch.randn(N, dtype=DT))
check("refactors at the first opportunity, not after refactor_every",
      p3.refactor() is True and p3._refactors == 1)


# ---------------------------------------------------------------------------
print("\n[8] step_params on SplaTAM's (1, k, num_frames) pose layout")
# The live code path. A shape or offset bug here would not crash - it would
# silently apply the rotation part of the step to translation, or write to the
# wrong frame's slice, and the run would look plausible.
NF, IDX = 20, 7
rot = torch.zeros(1, 4, NF, dtype=DT, requires_grad=True)
tra = torch.zeros(1, 3, NF, dtype=DT, requires_grad=True)
rot.grad = torch.zeros_like(rot)
tra.grad = torch.zeros_like(tra)
g_rot = torch.tensor([[1., 2., 3., 4.]], dtype=DT)
g_tra = torch.tensor([[5., 6., 7.]], dtype=DT)
rot.grad[..., IDX] = g_rot
tra.grad[..., IDX] = g_tra

pre = PosePreconditioner(dim=7, lr=0.002, tau=0.05, refactor_every=1,
                         carry=True, dtype=DT)
pre.refactor(force=True)                      # identity P on a cold M
pre.step_params([(rot, IDX), (tra, IDX)])

want = pre.P @ (torch.cat([g_rot.reshape(-1), g_tra.reshape(-1)]))
check("writes the concatenated step into the right slices",
      torch.allclose(rot[0, :, IDX], -0.002 * want[:4], atol=1e-12)
      and torch.allclose(tra[0, :, IDX], -0.002 * want[4:], atol=1e-12))

# Gauge projection: the step must carry NOTHING along the null direction.
rot2 = torch.zeros(1, 4, NF, dtype=DT, requires_grad=True)
tra2 = torch.zeros(1, 3, NF, dtype=DT, requires_grad=True)
rot2.grad = torch.zeros_like(rot2); tra2.grad = torch.zeros_like(tra2)
rot2.data[0, :, IDX] = torch.tensor([1., 0., 0., 0.], dtype=DT)
rot2.grad[..., IDX] = g_rot
tra2.grad[..., IDX] = g_tra
qv = rot2[0, :, IDX].detach()
gauge = torch.cat([qv / qv.norm(), torch.zeros(3, dtype=DT)])

pre2 = PosePreconditioner(dim=7, lr=0.002, tau=0.05, carry=False, dtype=DT)
before = torch.cat([rot2[0, :, IDX].detach().clone(), tra2[0, :, IDX].detach().clone()])
pre2.step_params([(rot2, IDX), (tra2, IDX)], gauge=gauge)
after = torch.cat([rot2[0, :, IDX].detach(), tra2[0, :, IDX].detach()])
applied = after - before
check("gauge component is removed from the applied step",
      abs(float(torch.dot(applied, gauge))) < 1e-15,
      f"<step, gauge> = {float(torch.dot(applied, gauge)):.3e}")
check("the rest of the step survives the projection",
      float(applied.norm()) > 0)

other = [j for j in range(NF) if j != IDX]
check("leaves every other frame untouched",
      bool(rot[..., other].abs().max() == 0) and bool(tra[..., other].abs().max() == 0))

check("M saw all 7 coordinates, not just the first block",
      bool(pre.M[6, 6] > 0) and bool(pre.M[0, 6].abs() > 0),
      f"M[0,6] = {float(pre.M[0, 6]):.4f}")


# ---------------------------------------------------------------------------
print("\n[10] STAGE 1: the tangent Jacobian, against finite differences")
# THE POINT OF THIS TEST. The Gauss-Newton line died because the RENDERER's
# backward disagreed with finite differences of the rendered loss at cos=0.868,
# flat across four decades of eps (record §4.6). This Jacobian is a different
# animal: a closed-form algebraic map between two parameterisations of the same
# pose. If it is right it agrees with FD to machine precision, and if it does
# not, it is wrong and fixable - there is no "the function is non-smooth"
# escape hatch here.
from utils.pose_preconditioner import (  # noqa: E402
    quat_trans_grad_to_tangent, apply_tangent_step, _quat_to_mat,
)
from utils.gn_tracking import mat_to_quat  # noqa: E402

# A pose well away from any branch boundary, and a LINEAR loss so dL/d(q,t) is
# exactly known and the only thing under test is the chain rule.
T0 = se3_exp(torch.tensor([0.3, -0.2, 0.5, 0.35, -0.2, 0.15], dtype=DT))
q0, t0 = mat_to_quat(T0[:3, :3]), T0[:3, 3]
q0 = q0 / q0.norm()
a = torch.tensor([0.7, -1.3, 0.4, 2.1], dtype=DT)   # dL/dq
b = torch.tensor([-0.9, 1.7, 0.6], dtype=DT)        # dL/dt


def L_of_xi(xi):
    qn, tn = apply_tangent_step(q0, t0, xi, mat_to_quat)
    # Sign-align: q and -q are the same rotation, so a raw dot product with a
    # fixed `a` would flip sign across the FD stencil and produce garbage.
    if float(torch.dot(qn, q0)) < 0:
        qn = -qn
    return float(torch.dot(a, qn) + torch.dot(b, tn))


g_xi = quat_trans_grad_to_tangent(q0, t0, a, b)
fd = torch.zeros(6, dtype=DT)
eps = 1e-6
for i in range(6):
    e = torch.zeros(6, dtype=DT); e[i] = eps
    fd[i] = (L_of_xi(e) - L_of_xi(-e)) / (2 * eps)

cos = float(torch.dot(g_xi, fd) / (g_xi.norm() * fd.norm()))
rel = float((g_xi - fd).norm() / fd.norm())
check("analytic tangent gradient matches central differences",
      rel < 1e-7, f"cos = {cos:.10f}, rel_err = {rel:.3e}")

# The V-curve the GN campaign never saw: on a genuinely smooth map, shrinking
# eps must reduce the error until roundoff takes over. A FLAT curve is what
# said FD and the renderer were measuring different functions.
errs = []
for e_ in (1e-3, 1e-4, 1e-5, 1e-6):
    fdx = torch.zeros(6, dtype=DT)
    for i in range(6):
        e = torch.zeros(6, dtype=DT); e[i] = e_
        fdx[i] = (L_of_xi(e) - L_of_xi(-e)) / (2 * e_)
    errs.append(float((g_xi - fdx).norm() / fdx.norm()))
print(f"        eps sweep 1e-3..1e-6: " + "  ".join(f"{e:.2e}" for e in errs))
check("error FALLS as eps shrinks (smooth map, unlike the render)",
      errs[-1] < errs[0] / 10)

# No gauge direction to leak into: dq/dtheta is orthogonal to q, so a gradient
# that points purely along q (pure gauge, invisible to the loss) must map to
# exactly zero tangent gradient.
g_gauge = quat_trans_grad_to_tangent(q0, t0, q0.clone(), torch.zeros(3, dtype=DT))
check("a pure-gauge gradient maps to ZERO tangent gradient",
      float(g_gauge.norm()) < 1e-15,
      f"|g_xi| = {float(g_gauge.norm()):.3e}  <- Stage 0's bug, absent by construction")

# Round trip: a step then its inverse returns the pose.
xi = torch.tensor([0.01, -0.02, 0.015, 0.008, -0.004, 0.006], dtype=DT)
q1, t1 = apply_tangent_step(q0, t0, xi, mat_to_quat)
q2, t2 = apply_tangent_step(q1, t1, -xi, mat_to_quat)
if float(torch.dot(q2, q0)) < 0:
    q2 = -q2
check("exp(-xi) undoes exp(xi)",
      torch.allclose(q2, q0, atol=1e-12) and torch.allclose(t2, t0, atol=1e-12),
      f"max|dq| = {(q2 - q0).abs().max():.3e}")


# ---------------------------------------------------------------------------
print("\n[11] the stopping reference must not reward divergence")
# THE SPIRAL THIS PREVENTS. |g|/|g_0| normalises by the frame's OWN starting
# gradient, so a diverged frame - whose |g_0| is enormous - satisfies a 20x
# drop almost immediately, quits at the floor while still wrong, commits the
# pose, and hands the next frame a worse start. Divergence makes the criterion
# fire SOONER. Observed on the VM as every frame stopping at exactly STOP_MIN
# with mapping down to 2 it/s from ~90.
pre_ref = PosePreconditioner(dim=6, lr=0.004, carry=True, dtype=DT)
torch.manual_seed(0)


def _run_frame(pre, scale, n=30):
    pre.carry_frame(transport=None)
    out = []
    for i in range(n):
        pre.step(torch.randn(6, dtype=DT) * scale * (0.85 ** i))
        pre.refactor()
        out.append((float(pre.rel_grad()), float(pre.rel_grad_ref())))
    return out


for _ in range(8):
    _run_frame(pre_ref, 1.0)                 # establish the running average
bad = _run_frame(pre_ref, 100.0)             # a frame that starts 100x wrong

THR = 0.05
f_stop = next((i for i, (a, _) in enumerate(bad) if a < THR), None)
r_stop = next((i for i, (_, b) in enumerate(bad) if b < THR), None)
check("per-frame reference quits early on a diverged frame (the bug)",
      f_stop is not None and f_stop < 20, f"quits at iter {f_stop}")
check("running reference refuses to quit on it",
      r_stop is None, f"stops at {r_stop} (None = runs to cap)")

# And it must NOT make healthy frames run forever - otherwise it just trades
# the failure for the loss of all the speedup.
ok = _run_frame(pre_ref, 1.0)
h_stop = next((i for i, (_, b) in enumerate(ok) if b < THR), None)
check("healthy frames still stop under the running reference",
      h_stop is not None, f"stops at iter {h_stop}")

# THE CASE THE FIRST VERSION OF THIS TEST MISSED, and the one that actually
# happened on the VM. A single diverged frame was checked against an
# ESTABLISHED reference and correctly refused. But a run that degrades
# GRADUALLY drags the reference up with it - each frame within the per-frame
# clamp, the cumulative drift unbounded - until the guard dissolves and every
# frame stops at the floor again. Rejecting outliers from the average instead
# of clamping them is what makes the reference a property of HEALTHY frames.
creep = PosePreconditioner(dim=6, lr=0.004, carry=True, dtype=DT)
for _ in range(10):
    _run_frame(creep, 1.0)
ema_healthy = float(creep._g0_ema)
for k in range(40):                       # 40 frames, each 15% worse
    _run_frame(creep, 1.0 * (1.15 ** (k + 1)))
check("a gradually degrading run cannot drag the reference with it",
      float(creep._g0_ema) <= 2.5 * ema_healthy,
      f"{ema_healthy:.3f} -> {float(creep._g0_ema):.3f} after 40 worsening frames")

# And the hard guarantee: a frame that STARTS anomalous is barred from stopping
# early at all, whatever the ratio does.
check("an anomalous frame is flagged and cannot stop early",
      float(creep.anomalous()) > 0.5,
      f"anomalous = {float(creep.anomalous())}")
_run_frame(creep, 1.0)
check("a healthy frame is not flagged", float(creep.anomalous()) < 0.5)


# ---------------------------------------------------------------------------
print("\n[12] the capturable helpers must equal the validated ones")
# CUDA graph capture records kernel launches once and replays them. Any Python
# branch taken during capture is FROZEN - the replay always follows the branch
# that happened to be recorded - and any host readback is a sync that cannot be
# captured at all. gn_tracking's se3_exp does `ang = float(theta.norm())` and
# branches on it; mat_to_quat branches four ways on trace comparisons. Both had
# to be rewritten branch-free, and both must agree with the originals or the
# tracker silently changes when the graph is switched on.
from utils.pose_preconditioner import (  # noqa: E402
    se3_exp_capturable, mat_to_quat_capturable,
)

worst_T = worst_q = 0.0
for _scale in (0.0, 1e-9, 1e-6, 1e-4, 1e-2, 0.1, 0.5):
    for _ in range(120):
        _xi = torch.randn(6, dtype=DT) * _scale
        _a, _b = se3_exp(_xi), se3_exp_capturable(_xi)
        worst_T = max(worst_T, float((_a - _b).abs().max()))
        _qa, _qb = mat_to_quat(_a[:3, :3]), mat_to_quat_capturable(_a[:3, :3])
        if float(torch.dot(_qa, _qb)) < 0:
            _qb = -_qb
        worst_q = max(worst_q, float((_qa - _qb).abs().max()))

check("se3_exp_capturable == se3_exp across the used range",
      worst_T < 1e-10, f"max|dT| = {worst_T:.3e}")
check("mat_to_quat_capturable == mat_to_quat across the used range",
      worst_q < 1e-7, f"max|dq| = {worst_q:.3e}")

# The closed forms are 0/0 at zero rotation. torch.where selects around it, but
# only if the denominators are clamped - otherwise the unselected NaN branch
# still poisons nothing in value terms yet signals the guard is missing.
_z = se3_exp_capturable(torch.zeros(6, dtype=DT))
check("finite and exactly identity at zero rotation",
      bool(torch.isfinite(_z).all()) and torch.allclose(_z, torch.eye(4, dtype=DT)))

# No host sync anywhere: every intermediate must stay a tensor. A float() would
# not raise here, so assert on the types the helpers produce.
_probe = se3_exp_capturable(torch.tensor([0.01, 0, 0, 0.02, 0, 0], dtype=DT))
check("returns a tensor, not a host-assembled matrix",
      isinstance(_probe, torch.Tensor) and _probe.shape == (4, 4))



# ---------------------------------------------------------------------------
print("\n[13] shrinkage interpolates between the full matrix and Adam")
# M is 6x6 symmetric - 21 parameters accumulated from RANK-1 updates - so at a
# 40-iteration budget it is a poor estimate, and inverting noisy off-diagonals
# can cost more than the coupling they capture is worth.
#
# It is also the CENTRAL CLAIM made measurable. Rotation covariance (test 4) is
# precisely the property a full matrix HAS and a diagonal does NOT, so this knob
# should visibly trade it away - which is what makes a `shrink` sweep evidence
# about the mechanism rather than just another hyperparameter.
_Msh = (Q * torch.tensor([1., .5, .2, .05, .02, .01], dtype=DT)) @ Q.T
_full = _inv_sqrt_damped(_Msh, 0.05, 1e-12, 1.0)
_diag = _inv_sqrt_damped(_Msh, 0.05, 1e-12, 0.0)
_half = _inv_sqrt_damped(_Msh, 0.05, 1e-12, 0.5)


def _offdiag(A):
    return float((A - torch.diag(torch.diagonal(A))).abs().max())


check("shrink=1 keeps the coupling",
      _offdiag(_full) > 0.1, f"max off-diag {_offdiag(_full):.4f}")
check("shrink=0 is exactly diagonal",
      _offdiag(_diag) < 1e-12, f"max off-diag {_offdiag(_diag):.2e}")
check("shrink=0 == preconditioning on diag(M) alone",
      torch.allclose(_diag,
                     _inv_sqrt_damped(torch.diag(torch.diagonal(_Msh)),
                                      0.05, 1e-12, 1.0), atol=1e-12))
check("shrink=0.5 sits between them",
      _offdiag(_diag) < _offdiag(_half) < _offdiag(_full),
      f"{_offdiag(_diag):.3f} < {_offdiag(_half):.3f} < {_offdiag(_full):.3f}")

# The property being traded, stated directly: a diagonal preconditioner is tied
# to the axes it was written in.
_R2, _ = torch.linalg.qr(torch.randn(6, 6, dtype=DT))
_rot_full = _inv_sqrt_damped(_R2.T @ _Msh @ _R2, 0.05, 1e-12, 1.0)
_rot_diag = _inv_sqrt_damped(_R2.T @ _Msh @ _R2, 0.05, 1e-12, 0.0)
check("shrink=1 transforms covariantly under a basis change",
      torch.allclose(_rot_full, _R2.T @ _full @ _R2, atol=1e-8),
      f"max|d| = {float((_rot_full - _R2.T @ _full @ _R2).abs().max()):.2e}")
check("shrink=0 does NOT - the property the knob trades away",
      not torch.allclose(_rot_diag, _R2.T @ _diag @ _R2, atol=1e-6),
      f"max|d| = {float((_rot_diag - _R2.T @ _diag @ _R2).abs().max()):.3f}")


# ---------------------------------------------------------------------------
print(chr(10) + "[14] the sample metric: one backward, many samples")
# THE LOAD-BEARING TEST OF THIS ARM, and it is deliberately an IDENTITY rather
# than a tolerance. per_gaussian_tangent_grads claims to decompose the SAME
# total gradient the shipped path computes, differing only in grouping. If that
# is true, summing the rows must reproduce quat_trans_grad_to_tangent exactly -
# and if it is false, no SLAM run could tell us, because a wrong preconditioner
# only costs convergence rate and would come back as "a bit slower, maybe".
#
# The setup reproduces SplaTAM's ISOTROPIC tracking path exactly: the pose
# reaches the loss through transformed_pts alone, via F.normalize(q) and
# build_rotation, with the Gaussians detached.
_q0 = torch.randn(4, dtype=DT)
_q0 = _q0 / _q0.norm()
_qp = _q0.clone().requires_grad_(True)
_tp = torch.randn(3, dtype=DT).requires_grad_(True)
_pw = torch.randn(200, 3, dtype=DT) * 2.0          # world-frame centres
# An arbitrary smooth scalar of the camera-frame centres, standing in for the
# render. Nothing about the identity depends on which one - it is a statement
# about the parameterisation, not about the loss.
_w = torch.randn(200, 3, dtype=DT)

_qn = _qp / _qp.norm()                             # SplaTAM's F.normalize
_R = _quat_to_mat(_qn)
_pc = _pw @ _R.transpose(0, 1) + _tp               # = (R p_w + t) per row
_pc.retain_grad()
_loss = (torch.sin(_pc) * _w).sum()
_loss.backward()

_g_ref = quat_trans_grad_to_tangent(_q0, _tp.detach(), _qp.grad, _tp.grad)
_G = per_gaussian_tangent_grads(_pc.detach(), _pc.grad)
_g_sum = _G.sum(0)

check("per-Gaussian rows sum to the shipped tangent gradient",
      torch.allclose(_g_sum, _g_ref, atol=1e-10),
      f"max|d| = {float((_g_sum - _g_ref).abs().max()):.2e}")
check("  and it is not vacuous - the gradient is not ~0",
      float(_g_ref.norm()) > 1e-3, f"|g| = {float(_g_ref.norm()):.4f}")
check("G has one row per Gaussian", tuple(_G.shape) == (200, 6), str(tuple(_G.shape)))

# THE POINT OF THE WHOLE ARM. The shipped estimator needs >= 6 iterations
# before M can even be full rank; this is full rank from the first backward.
_Ms = sample_metric(_G, trace_match=True)
_M1 = torch.outer(_g_ref, _g_ref)
check("one backward gives a FULL-RANK metric",
      int(torch.linalg.matrix_rank(_Ms, tol=1e-10)) == 6,
      f"rank {int(torch.linalg.matrix_rank(_Ms, tol=1e-10))}")
check("  where the shipped rank-1 update gives rank 1",
      int(torch.linalg.matrix_rank(_M1, tol=1e-10)) == 1,
      f"rank {int(torch.linalg.matrix_rank(_M1, tol=1e-10))}")

# Trace matching is what keeps PRE_LR, tau and max_step_mult meaning what they
# meant. Untraced, this ratio is ~N and every step would shrink by ~sqrt(N).
check("trace-matched to |g|^2, so the step scale is unchanged",
      abs(float(torch.diagonal(_Ms).sum() - _g_ref.dot(_g_ref))) < 1e-8,
      f"tr(M_s) = {float(torch.diagonal(_Ms).sum()):.6f}, "
      f"|g|^2 = {float(_g_ref.dot(_g_ref)):.6f}")
_raw = sample_metric(_G, trace_match=False)
_neff = float(effective_samples(_G))
# THE RESCALING FACTOR IS n_eff, NOT N. This check originally asserted >10x on
# the reasoning that trace(G^T G) ~ N |g|^2, and it FAILED at 1.6 - because
# n_eff measures INCOHERENCE among the per-Gaussian gradients, not how many
# there are. Left as an explicit assertion of the true relation, since the
# false one was plausible enough to be written down twice.
check("  untraced scale is exactly n_eff, which is NOT N",
      abs(float(torch.diagonal(_raw).sum() / torch.diagonal(_Ms).sum()) - _neff)
      < 1e-8 and _neff < 200,
      f"n_eff = {_neff:.2f} from N = {_G.shape[0]} Gaussians")
check("  trace matching changes scale only, never shape",
      torch.allclose(_raw / torch.diagonal(_raw).sum(),
                     _Ms / torch.diagonal(_Ms).sum(), atol=1e-12))
check("sample metric is symmetric PSD",
      torch.allclose(_Ms, _Ms.T, atol=1e-12)
      and float(torch.linalg.eigvalsh(_Ms).min()) > -1e-12,
      f"min eig {float(torch.linalg.eigvalsh(_Ms).min()):.2e}")

# It must carry information a rank-1 update structurally cannot: the two
# metrics must not be proportional.
_cos = float((_Ms * _M1).sum() / (_Ms.norm() * _M1.norm()))
check("it is NOT a rescaled g g^T - it carries new directions",
      _cos < 0.99, f"cos(M_sample, g g^T) = {_cos:.4f}")
# THE GO/NO-GO NUMBER, exported so the run reports the same quantity the test
# asserts. n_eff was briefly given this job and it was the wrong quantity: it
# measures CANCELLATION (~1 incoherent, ~1/N fully aligned), not whether the
# metric's shape differs from rank 1.
check("  rank1_alignment computes exactly that, and is scale-free",
      abs(float(rank1_alignment(_G)) - _cos) < 1e-12
      and abs(float(rank1_alignment(_G * 7.0)) - _cos) < 1e-12,
      f"{float(rank1_alignment(_G)):.4f}")
# n_eff on INCOHERENT samples is ~1, not ~N. Pinned because the docstring said
# ~N for one commit and a stop rule was built on it.
check("  n_eff ~ 1 on incoherent samples, NOT ~N",
      0.2 < _neff < 5.0, f"n_eff = {_neff:.2f} at N = {_G.shape[0]}")
# And the degenerate case the stop rule actually cares about: identical rows.
_Gsame = _G[:1].expand(_G.shape[0], 6).contiguous()
check("  identical rows give n_eff ~ 1/N and alignment ~ 1",
      abs(float(effective_samples(_Gsame)) - 1.0 / _G.shape[0]) < 1e-9
      and float(rank1_alignment(_Gsame)) > 0.999,
      f"n_eff = {float(effective_samples(_Gsame)):.2e}, "
      f"cos = {float(rank1_alignment(_Gsame)):.4f}")

# And step() must actually consume it.
_pre = PosePreconditioner(dim=6, lr=1e-3, beta2=0.9, sample_metric=True,
                          refactor_every=1, dtype=DT)
_pre.carry_frame(transport=None)
_pre.step(_g_ref, M_inst=_Ms)
_pre.refactor(force=True)
check("step(M_inst=...) reaches M and is counted",
      _pre.sampled == 1
      and int(torch.linalg.matrix_rank(_pre.M, tol=1e-12)) == 6,
      f"sampled={_pre.sampled}, rank(M)={int(torch.linalg.matrix_rank(_pre.M, tol=1e-12))}")
_pre2 = PosePreconditioner(dim=6, lr=1e-3, beta2=0.9, dtype=DT)
_pre2.carry_frame(transport=None)
_pre2.step(_g_ref)
check("  while the default path is untouched (rank 1, not counted)",
      _pre2.sampled == 0
      and int(torch.linalg.matrix_rank(_pre2.M, tol=1e-12)) == 1,
      f"sampled={_pre2.sampled}, rank(M)={int(torch.linalg.matrix_rank(_pre2.M, tol=1e-12))}")



# ---------------------------------------------------------------------------
print(chr(10) + "[15] cold-start: no metric means a NORMALISED step, every frame")
# THE BUG THIS PINS. The no-metric branch used to be keyed on `_refactors == 0`,
# a GLOBAL counter, so it fired only at the very start of a RUN. With carry
# OFF, carry_frame resets P to the identity at the start of EVERY frame and
# refactor() runs AFTER step(), so the first step of every frame was raw
# gradient descent -lr*m, clamped by the trust region and therefore pinned
# EXACTLY at max_step_mult * lr * sqrt(n).
#
# Found in a real run by that exact signature: max |d| = 9.80e-02 against
# 10 * 0.004 * sqrt(6) = 9.7980e-02, where carry-on runs sit at 5.2-7.9e-02 and
# never reach the cap.
_LR, _MULT = 0.004, 10.0
_cap = _MULT * _LR * math.sqrt(6)
_pre15 = PosePreconditioner(dim=6, lr=_LR, max_step_mult=_MULT, carry=False,
                            refactor_every=1, dtype=DT)
# A large gradient, as a render summed over ~300k pixels produces.
_gbig = torch.randn(6, dtype=DT) * 1e3


def _first_step_norm(pre):
    """|delta| of the FIRST step of a fresh frame."""
    pre.carry_frame(transport=None)
    return float(pre.step(_gbig).norm())


_d0 = _first_step_norm(_pre15)
check("first step of a cold frame is Adam-magnitude, not the trust-region cap",
      abs(_d0 - _LR * math.sqrt(6)) < 1e-12,
      f"|d| = {_d0:.4e}, cap would be {_cap:.4e}")
check("  and it does NOT scale with |g| - that is the point of normalising",
      abs(_first_step_norm(PosePreconditioner(
          dim=6, lr=_LR, max_step_mult=_MULT, carry=False, refactor_every=1,
          dtype=DT)) - _d0) < 1e-12)

# After a refactor the frame HAS a metric, and the normalised branch must stop
# firing - otherwise the preconditioner would never actually be applied.
_pre15.refactor(force=True)
_d1 = float(_pre15.step(_gbig).norm())
check("  once refactored, the metric is applied and the step is bounded",
      _d1 <= _cap + 1e-12 and abs(_d1 - _LR * math.sqrt(6)) > 1e-12,
      f"|d| = {_d1:.4e} <= cap {_cap:.4e}")

# And the next frame goes cold again, because carry is off.
check("  the next cold frame is normalised again (per-FRAME, not per-run)",
      abs(_first_step_norm(_pre15) - _LR * math.sqrt(6)) < 1e-12)

# With carry ON the metric survives the frame boundary, so the normalised
# branch must NOT fire - that would throw away the carry.
_prec = PosePreconditioner(dim=6, lr=_LR, max_step_mult=_MULT, carry=True,
                           refactor_every=1, dtype=DT)
_prec.carry_frame(transport=None)
_prec.step(_gbig)
_prec.refactor(force=True)
_prec.carry_frame(transport=None)          # new frame, M carried
check("carry ON keeps its metric across the frame boundary",
      _prec._have_metric,
      "normalised branch would discard the carry")


# ---------------------------------------------------------------------------
print(chr(10) + "[16] isotropic frame-start prior - the clean no-carry comparison")
# WHY THIS EXISTS. A carry-off comparison starting from M = 0 does not isolate
# what it claims to. The rank-1 arm is SINGULAR for its first n steps and leans
# on the Levenberg floor; the sample arm is full rank from its first backward.
# So the arms differ in RANK, and rank is not the variable under test - SHAPE
# is. Seeding both with the same full-rank, correctly-scaled isotropic prior
# leaves shape as the only difference.
_g16 = torch.randn(6, dtype=DT) * 37.0


def _mk(iso, sample):
    p = PosePreconditioner(dim=6, lr=0.004, carry=False, m0_iso=iso,
                           sample_metric=sample, refactor_every=1, dtype=DT)
    p.carry_frame(transport=None)
    return p


# One backward's worth of per-Gaussian samples whose SUM is _g16, so both arms
# see the identical total gradient - the invariant group 14 established.
_G16 = torch.randn(500, 6, dtype=DT)
_G16 = _G16 - _G16.mean(0) + _g16 / 500.0
_M16 = sample_metric(_G16, trace_match=True)

_r1 = _mk(True, False); _r1.step(_g16)
_pg = _mk(True, True);  _pg.step(_g16, M_inst=_M16)

check("both arms start full rank with an isotropic prior",
      torch.linalg.matrix_rank(_r1.M, tol=1e-10) == 6
      and torch.linalg.matrix_rank(_pg.M, tol=1e-10) == 6,
      f"rank1 {int(torch.linalg.matrix_rank(_r1.M, tol=1e-10))}, "
      f"sample {int(torch.linalg.matrix_rank(_pg.M, tol=1e-10))}")
check("  where from M=0 the rank-1 arm is singular and the sample arm is not",
      torch.linalg.matrix_rank(_mk(False, False).M.clone().addr_(_g16, _g16),
                               tol=1e-10) == 1,
      "rank 1 - the confound this removes")
check("  and both start at the SAME scale, so only shape differs",
      abs(float(torch.diagonal(_r1.M).sum() - torch.diagonal(_pg.M).sum()))
      < 1e-8 * float(torch.diagonal(_r1.M).sum()),
      f"tr rank1 {float(torch.diagonal(_r1.M).sum()):.4e}, "
      f"tr sample {float(torch.diagonal(_pg.M).sum()):.4e}")
check("  but they are NOT the same matrix - shape is what is left",
      not torch.allclose(_r1.M, _pg.M, atol=1e-10),
      f"max|d| = {float((_r1.M - _pg.M).abs().max()):.3e}")

# The seed must carry the trace-matching convention exactly: tr(M_0) = |g_0|^2.
_seed = _mk(True, False)
_seed.step(_g16)
_want = float(_g16.dot(_g16))
_beta2 = _seed.beta2
# M = beta2 * (|g|^2/n) I + (1-beta2) g g^T  ->  trace = |g|^2 either way.
check("  the isotropic seed preserves trace(M) = |g_0|^2",
      abs(float(torch.diagonal(_seed.M).sum()) - _want) < 1e-8 * _want,
      f"tr = {float(torch.diagonal(_seed.M).sum()):.6e} vs |g|^2 = {_want:.6e}")
check("  and it disables the cold-EMA bias correction, which would double-count",
      _seed._M_cold is False,
      "M starts at the right magnitude, so 1/(1-beta2^t) must not apply")


# ---------------------------------------------------------------------------
print(chr(10) + "[17] damped BFGS - curvature from the steps already taken")
# THE SECANT IDENTITY IS THE LOAD-BEARING ONE, and like group 14 it is an
# IDENTITY rather than a tolerance. For an exactly quadratic objective
# g = H xi + b, one BFGS update from a single pair must invert H along that
# direction exactly: B y = s. If that fails the update is simply wrong, and a
# preconditioner would hide it as "converges a bit slower".
_H17 = torch.tensor([[4.0, 1.0, 0, 0, 0, 0],
                     [1.0, 3.0, 0, 0, 0, 0],
                     [0, 0, 2.0, 0.5, 0, 0],
                     [0, 0, 0.5, 1.0, 0, 0],
                     [0, 0, 0, 0, 5.0, 0],
                     [0, 0, 0, 0, 0, 0.25]], dtype=DT)
_s17 = torch.randn(6, dtype=DT)
_y17 = _H17 @ _s17                      # exact secant pair for this quadratic
_B17 = bfgs_update(torch.eye(6, dtype=DT), _s17, _y17)

check("satisfies the secant equation B y = s exactly",
      torch.allclose(_B17 @ _y17, _s17, atol=1e-12),
      f"max|d| = {float((_B17 @ _y17 - _s17).abs().max()):.2e}")
check("  stays symmetric",
      torch.allclose(_B17, _B17.T, atol=1e-12))
check("  stays positive definite",
      float(torch.linalg.eigvalsh(_B17).min()) > 0,
      f"min eig {float(torch.linalg.eigvalsh(_B17).min()):.3e}")

# THE SCREEN. A pair with non-positive curvature along s carries no usable
# information and would destroy positive-definiteness - the property the
# fixed-point argument depends on. It must leave B untouched, not "damp" it.
_B0 = torch.randn(6, 6, dtype=DT)
_B0 = _B0 @ _B0.T + torch.eye(6, dtype=DT)
check("rejects a negative-curvature pair, leaving B untouched",
      torch.equal(bfgs_update(_B0, _s17, -_y17), _B0),
      "y^T s < 0")
check("  rejects an orthogonal pair too (y^T s == 0)",
      torch.equal(bfgs_update(_B0, _s17,
                              torch.linalg.cross(_s17[:3], torch.randn(3, dtype=DT))
                              .repeat(2) * 0.0 + _s17.roll(3) * 0.0), _B0),
      "zero y")
check("  and does not emit inf/nan on a rejected pair",
      torch.isfinite(bfgs_update(_B0, _s17, -_y17)).all(),
      "the reciprocal is guarded, not just its result")

# WHAT ACTUALLY MATTERS: driven as an optimiser it converges superlinearly.
#
# A CHECK ASSERTING `B -> H^-1` WAS WRITTEN HERE FIRST AND WAS WRONG. That is a
# textbook property of BFGS along CONJUGATE directions; from arbitrary ones each
# rank-2 update enforces the newest secant equation and generally breaks the
# previous ones, so B need not approach H^-1 at all. Measured below: |x| falls
# by six orders of magnitude while |B - H^-1| stays ~0.15. BFGS does not need
# the inverse Hessian - it needs to be right along the directions it searches,
# which is exactly why it works as a PRECONDITIONER rather than a solver.
_Hinv = torch.linalg.inv(_H17)
_Bo = torch.eye(6, dtype=DT)
_xo = torch.ones(6, dtype=DT)
for _ in range(12):
    _go = _H17 @ _xo
    _so = -(_Bo @ _go)
    _xn = _xo + _so
    _Bo = bfgs_update(_Bo, _so, _H17 @ _xn - _go)
    _xo = _xn
_xg = torch.ones(6, dtype=DT)
for _ in range(12):
    _xg = _xg - 0.35 * (_H17 @ _xg)
check("as an optimiser it converges superlinearly, far past GD",
      float(_xo.norm()) < 1e-4 * float(_xg.norm()),
      f"|x| BFGS {float(_xo.norm()):.2e} vs GD {float(_xg.norm()):.2e} at 12 steps")
check("  and it does NOT need B = H^-1 to do it",
      float((_Bo - _Hinv).abs().max()) > 1e-2,
      f"|B - H^-1| = {float((_Bo - _Hinv).abs().max()):.2e} while |x| = "
      f"{float(_xo.norm()):.2e}")

# Spectrum clamping bounds the step without changing the direction structure.
_Bbad = torch.diag(torch.tensor([1e6, 1.0, 1.0, 1.0, 1.0, 1e-6], dtype=DT))
_Bc = clamp_spectrum(_Bbad, 20.0)
_ev = torch.linalg.eigvalsh(_Bc)
check("clamp_spectrum bounds the condition number",
      float(_ev.max() / _ev.min()) <= 20.0 + 1e-9,
      f"cond {float(_ev.max() / _ev.min()):.2f} <= 20")
check("  and symmetrises drift",
      torch.allclose(_Bc, _Bc.T, atol=1e-14))

# Wiring: the pair must NOT cross a frame boundary.
_pb = PosePreconditioner(dim=6, lr=0.004, bfgs=True, refactor_every=1, dtype=DT)
_pb.carry_frame(transport=None)
_pb.step(torch.randn(6, dtype=DT))
check("a pair is available after one step",
      _pb._have_pair)
_pb.carry_frame(transport=None)
check("  but NEVER across a frame boundary",
      not _pb._have_pair,
      "the pose jumped and the image changed - y would describe neither")
_pb.step(torch.randn(6, dtype=DT))
_pb.note_objective_change()
check("  nor across a sparse/dense phase flip",
      not _pb._have_pair,
      "the tile mask changed, so the objective did")


# ---------------------------------------------------------------------------
print(chr(10) + "[18] BFGS must SELF-SCALE - the step has to decay on approach")
# THE BUG THIS PINS, found in a real run. Running BFGS with target_step_norm
# normalises every step to the same length, measured as mean |d| == max |d| ==
# 1.00x Adam over 2490 steps. Harmless for M, whose step is scale-free anyway;
# fatal for BFGS, because -B g with B ~ H^-1 is a Newton step that SHRINKS as
# the gradient shrinks. Fix its length and the optimiser orbits forever.
_LR18 = 1.0
_pa = PosePreconditioner(dim=6, lr=_LR18, bfgs=True, carry=False,
                         refactor_every=1, beta1=0.0, dtype=DT)
_pa.carry_frame(transport=None)
_H18 = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5, 0.25], dtype=DT))
_x = torch.ones(6, dtype=DT)
_norms = []
for _ in range(40):
    _d = _pa.step(_H18 @ _x)
    _pa.refactor(force=True)
    _x = _x + _d
    _norms.append(float(_d.norm()))

check("the step DECAYS as the gradient does",
      _norms[-1] < 0.05 * _norms[0],
      f"|d| {_norms[0]:.3e} -> {_norms[-1]:.3e}")
check("  so the iterate actually converges",
      float(_x.norm()) < 1e-3, f"|x| = {float(_x.norm()):.3e}")
# The failure mode, reproduced deliberately so the contrast is measured and not
# asserted: with target_step_norm every step is the SAME length forever.
_pf = PosePreconditioner(dim=6, lr=_LR18, bfgs=True, carry=False,
                         target_step_norm=4.9e-3, refactor_every=1,
                         beta1=0.0, dtype=DT)
_pf.carry_frame(transport=None)
_x2 = torch.ones(6, dtype=DT)
_n2 = []
for _ in range(40):
    _d = _pf.step(_H18 @ _x2)
    _pf.refactor(force=True)
    _x2 = _x2 + _d
    _n2.append(float(_d.norm()))
check("  where target_step_norm pins every step to one length",
      abs(max(_n2) - min(_n2)) < 1e-12,
      f"mean == max == {_n2[-1]:.3e}, exactly the real run's signature")

# Autoscale must make the FIRST step Adam-magnitude despite B_0's arbitrary
# scale and a gradient of any size - otherwise -B_0 g is ~|g|, which on a loss
# summed over ~300k pixels destroys the pose.
for _gs in (1e-2, 1.0, 1e4):
    _ps = PosePreconditioner(dim=6, lr=0.004, bfgs=True, carry=False,
                             refactor_every=1, beta1=0.0, dtype=DT)
    _ps.carry_frame(transport=None)
    _g = torch.randn(6, dtype=DT) * _gs
    _d0 = float(_ps.step(_g).norm())
    check(f"  first step is Adam-magnitude at |g|~{_gs:g}",
          abs(_d0 - 0.004 * math.sqrt(6)) < 1e-10,
          f"|d| = {_d0:.4e}")
    # THE HOLE THE ORIGINAL VERSION OF THIS CHECK LEFT. The FIRST step of a
    # frame takes the no-metric fallback and never touches B, so testing it
    # alone passed while B was scaled by the wrong factor. The SECOND step is
    # the first one B actually produces, and it was coming out at lr * ref -
    # 500x too small - which made BFGS inert and cost a full sweep to find.
    _ps.refactor(force=True)
    _d1 = float(_ps.step(_g).norm())
    check(f"    and so is the FIRST B-DRIVEN step at |g|~{_gs:g}",
          0.1 * 0.004 * math.sqrt(6) < _d1 < 10.0 * 0.004 * math.sqrt(6),
          f"|d| = {_d1:.4e} vs ref {0.004 * math.sqrt(6):.4e}")


# ---------------------------------------------------------------------------
print(chr(10) + "[19] the curvature screen is an ANGLE, and B's growth is bounded")
# WHY BOTH GUARDS EXIST, from a real run at i20: pairs 599/882 accepted and
# max |d| = 4.90e-02 = exactly 10 * 0.002 * sqrt(6), the trust-region cap. B had
# grown ~10x inside a frame and the clamp, not the metric, was setting the step
# length. Two separate holes:
#   - curv_eps = 1e-8 is not a threshold. It accepts any pair with positive
#     curvature, including near-flat ones where rho = 1/(y^T s) is enormous.
#   - clamp_spectrum bounds cond(B) but NOT its magnitude.
_s19 = torch.randn(6, dtype=DT)
_s19 = _s19 / _s19.norm()
# A pair that is 'positive' but almost orthogonal: real curvature signal ~1e-4.
_perp = torch.linalg.svd(_s19.unsqueeze(0))[2][1:].sum(0)
_y_flat = _perp / _perp.norm() + 1e-4 * _s19
_B19 = torch.eye(6, dtype=DT)

check("a near-flat pair is REJECTED at eps=1e-2",
      torch.equal(bfgs_update(_B19, _s19, _y_flat, 1e-2), _B19),
      f"cos(y,s) = {float(_y_flat.dot(_s19) / _y_flat.norm()):.2e}")
check("  and ACCEPTED at eps=1e-8 - the hole this closes",
      not torch.equal(bfgs_update(_B19, _s19, _y_flat, 1e-8), _B19))
check("  where accepting it inflates B enormously",
      float(bfgs_update(_B19, _s19, _y_flat, 1e-8).norm())
      > 100.0 * float(_B19.norm()),
      f"|B| {float(_B19.norm()):.2f} -> "
      f"{float(bfgs_update(_B19, _s19, _y_flat, 1e-8).norm()):.2e}")
# A healthy pair must still pass, or the screen is just an off switch.
check("  while a well-conditioned pair still passes",
      not torch.equal(bfgs_update(_B19, _s19, _H17 @ _s19, 1e-2), _B19),
      f"cos(y,s) = "
      f"{float((_H17 @ _s19).dot(_s19) / (_H17 @ _s19).norm()):.3f}")

# clamp_spectrum bounds the RATIO, not the SCALE - which is why the growth
# bound is a separate guard rather than a tighter max_cond.
_Bbig = 500.0 * torch.eye(6, dtype=DT)
_Bcl = clamp_spectrum(_Bbig, 20.0)
check("clamp_spectrum leaves magnitude untouched - hence bfgs_max_growth",
      abs(float(_Bcl.norm()) - float(_Bbig.norm())) < 1e-9,
      f"|B| {float(_Bbig.norm()):.1f} -> {float(_Bcl.norm()):.1f}, cond already 1")


# ---------------------------------------------------------------------------
print(chr(10) + "[20] B cannot overflow between refactors, and eigh is guarded")
# THE CRASH THIS PINS: linalg.eigh "failed to converge ... ill-conditioned" at
# frame 225 of 250, killing the run. V = I - rho s y^T has norm ~1/cos(y,s), so
# a pair AT the screen boundary (cos = curv_eps) grows B by ~1/curv_eps^2 in one
# update. The growth bound lived only in refactor(), which rebuilds every 10
# iterations, so B could compound past float32 range inside that window.
_sb = torch.randn(6, dtype=DT)
_sb = _sb / _sb.norm()
# A pair sitting exactly on the boundary - the worst case the screen admits.
_perp_b = torch.linalg.svd(_sb.unsqueeze(0))[2][1:].sum(0)
_perp_b = _perp_b / _perp_b.norm()
_yb = _perp_b + 1.01e-2 * _sb
_Bg = torch.eye(6, dtype=DT)
for _ in range(10):
    _Bg = bfgs_update(_Bg, _sb, _yb, 1e-2)
# Measured 4.0e3x over one refactor window, which compounds across windows -
# that is what reached float32 range at frame 225. A first version of this
# check asserted >1e6 in ten updates and failed at 9.9e3: the growth is
# per-WINDOW and compounding, not explosive in ten steps, and the assertion
# should say what was measured rather than what sounded alarming.
check("an at-boundary pair grows B by orders of magnitude per refactor window",
      float(_Bg.norm()) > 1e3 * 2.45,
      f"|B| 2.45 -> {float(_Bg.norm()):.2e} in 10 updates, unclamped "
      f"({float(_Bg.norm()) / 2.45:.0e}x per window)")

# The per-iteration bound must hold it, whatever the pairs do.
_pg = PosePreconditioner(dim=6, lr=0.002, bfgs=True, carry=False,
                         refactor_every=10, beta1=0.0, bfgs_max_growth=4.0,
                         dtype=DT)
_pg.carry_frame(transport=None)
for _k in range(40):
    _pg.step(torch.randn(6, dtype=DT) * 1e3)
    _pg.refactor()
check("  but the per-iteration bound holds B finite across 40 steps",
      bool(torch.isfinite(_pg.B).all())
      and float(_pg.B.norm()) <= 4.0 * float(_pg._B_scale0) * (1 + 1e-6),
      f"|B| = {float(_pg.B.norm()):.3e}, limit "
      f"{4.0 * float(_pg._B_scale0):.3e}")

# And the last line of defence: a non-finite B must not reach eigh at all.
_Bnan = torch.eye(6, dtype=DT) * float("inf")
check("clamp_spectrum returns identity on a non-finite B instead of raising",
      torch.allclose(clamp_spectrum(_Bnan, 20.0), torch.eye(6, dtype=DT)),
      "cuSOLVER raises on inf/nan - a dead run, not a bad number")


# ---------------------------------------------------------------------------
print(chr(10) + "[21] plateau stopping - monotone where the other criteria are not")
# THE THREE MEASURED FAILURES THIS REPLACES:
#   loss_eps  elbow at iteration 5, tail_mass 1.00 - the loss stops carrying
#             information long before the pose stops moving.
#   rel_grad  NON-MONOTONE. Real curve at it 0/1/2/5/10/20/30/40:
#             1.000 0.847 1.043 0.768 0.877 0.843 0.684 0.619
#   pose_eps  scale-free step, never decays, ships as 0.0.
# _stall counts iterations since |g| last set a new frame minimum, so it is
# monotone by construction.
_REAL = [1.000, 0.847, 1.043, 0.768, 0.877, 0.843, 0.684, 0.619]
_p21 = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_p21.carry_frame(transport=None)
_e = torch.zeros(6, dtype=DT)
_e[0] = 1.0
_stalls = []
for _v in _REAL:
    _p21.step(_e * _v)
    _stalls.append(float(_p21.stalled()))

check("an overshoot spike above |g_0| does NOT reset progress",
      _stalls[2] > _stalls[1],
      f"it2 (|g|=1.043) stall {_stalls[2]:.0f} > it1 stall {_stalls[1]:.0f}")
check("  a new minimum DOES reset it",
      _stalls[3] == 0.0,
      f"it3 (|g|=0.768, a new best) stall {_stalls[3]:.0f}")
check("  and a value above the running best does not",
      _stalls[4] == 1.0 and _stalls[5] == 2.0,
      f"it4 {_stalls[4]:.0f}, it5 {_stalls[5]:.0f} - 0.877 and 0.843 both "
      f"above the best 0.768")
check("  while 0.684 and 0.619 are new bests and reset again",
      _stalls[6] == 0.0 and _stalls[7] == 0.0,
      f"it6 {_stalls[6]:.0f}, it7 {_stalls[7]:.0f}")

# A true plateau must actually fire, or the criterion is just an off switch.
for _ in range(12):
    _p21.step(_e * 0.62)          # the measured plateau value
check("a genuine plateau raises the stall count past any patience",
      float(_p21.stalled()) >= 10,
      f"stall = {float(_p21.stalled()):.0f} after 12 flat iterations")

# The improve margin must be a REAL margin: 0.999x is not progress.
_p22 = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_p22.carry_frame(transport=None)
for _k in range(15):
    _p22.step(_e * (1.0 - 0.001 * _k))     # 0.1% better each time
check("  drifting by less than the margin counts as stalled, not progress",
      float(_p22.stalled()) >= 10,
      f"stall = {float(_p22.stalled()):.0f} on a 0.1%/iter drift at margin 1%")

# It must reset across frames - a plateau is a property of one frame.
_p21.carry_frame(transport=None)
check("  and the counter resets at a frame boundary",
      float(_p21.stalled()) == 0.0 and not torch.isfinite(_p21._g_best))


# ---------------------------------------------------------------------------
print(chr(10) + "[22] the progress gate - a diverging frame must NOT stop early")
# THE DEATH SPIRAL THIS CLOSES. A diverging frame does not improve, so the raw
# stall counter climbs immediately, hits any patience at the STOP_MIN floor, and
# the frame quits while still wrong - committing a bad pose so the next frame
# starts worse. Observed with the anomaly guard ON: frames past the divergence
# point ran 6-12 iterations against ~25 for healthy ones, and keyframe selection
# collapsed from 18 entries to 2 within three frames.
_e22 = torch.zeros(6, dtype=DT)
_e22[0] = 1.0

# A DIVERGING frame: the gradient never beats its start.
_pd = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_pd.carry_frame(transport=None)
for _v in [1.0] + [1.2, 1.5, 1.4, 1.8, 2.1, 1.9, 2.4, 2.2, 2.7, 3.0,
                   3.3, 2.9, 3.6, 4.0]:
    _pd.step(_e22 * _v)
check("a frame that never beats its start reports 0 stall, forever",
      float(_pd.stalled()) == 0.0,
      f"raw counter is {float(_pd._stall):.0f}, reported {float(_pd.stalled()):.0f}")
check("  so no patience can ever fire on it",
      float(_pd.stalled()) < 10,
      "it runs its full budget instead of quitting at the floor")

# A CONVERGING frame must be unaffected - it beats its start almost at once.
_pc = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_pc.carry_frame(transport=None)
for _v in [1.0, 0.80, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74,
           0.74, 0.74, 0.74, 0.74, 0.74]:
    _pc.step(_e22 * _v)
check("a converging frame still stalls and stops normally",
      float(_pc.stalled()) >= 10,
      f"stall = {float(_pc.stalled()):.0f} after plateauing at 0.74")

# The boundary: improving by LESS than the margin is not progress.
_pb = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_pb.carry_frame(transport=None)
for _v in [1.0] + [0.999] * 14:
    _pb.step(_e22 * _v)
check("  and a 0.1% improvement on the start is NOT progress at a 1% margin",
      float(_pb.stalled()) == 0.0,
      f"reported {float(_pb.stalled()):.0f}, raw {float(_pb._stall):.0f}")


# ---------------------------------------------------------------------------
print(chr(10) + "[23] the plateau tracker survives a handoff")
# THE BUG THIS PINS. step() runs only for the preconditioner's share of a
# frame, so every quantity it maintains - _g0, _g_best, the stall counter -
# FREEZES the moment Adam takes over. A frame that had not already stalled by
# the handoff iteration could therefore never stop, and ran to the cap:
# observed as roughly half the frames at exactly 200/200 while the rest stopped
# at 14-21. observe() is the bookkeeping half, called during the Adam phase.
_e23 = torch.zeros(6, dtype=DT)
_e23[0] = 1.0
_ph = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_ph.carry_frame(transport=None)
# 20 preconditioner iterations, still improving at the handoff.
for _k, _v in enumerate([1.0] + [0.9 - 0.01 * _i for _i in range(19)]):
    _ph.step(_e23 * _v)
check("still improving at the handoff, so nothing has stalled yet",
      float(_ph.stalled()) == 0.0,
      f"stall = {float(_ph.stalled()):.0f}, g_best = {float(_ph._g_best):.3f}")

# Adam phase: observe() only, gradient now plateaued.
for _ in range(15):
    _ph.observe(_e23 * 0.72)
check("  observe() keeps the counter alive so the criterion CAN fire",
      float(_ph.stalled()) >= 10,
      f"stall = {float(_ph.stalled()):.0f} after 15 Adam-phase iterations")

# Without observe() the counter would be frozen - the failure, reproduced.
_pf = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
_pf.carry_frame(transport=None)
for _v in [1.0] + [0.9 - 0.01 * _i for _i in range(19)]:
    _pf.step(_e23 * _v)
_frozen = float(_pf.stalled())
check("  and WITHOUT it the counter is frozen - the frame could never stop",
      _frozen == 0.0,
      f"stall stays {_frozen:.0f} however long the Adam phase runs")

# observe() must not take a step or touch the metric.
_M_before = _ph.M.clone()
_m_before = _ph.m.clone()
_ph.observe(_e23 * 0.72)
check("observe() does NOT update the metric or the first moment",
      torch.equal(_ph.M, _M_before) and torch.equal(_ph.m, _m_before),
      "bookkeeping only - the Adam phase owns the step")

# ---------------------------------------------------------------------------
print(chr(10) + "[24] an EMPTY RENDER cannot make a frame immortal")
# THE BUG THIS PINS, and it cost a full GSLAM sweep. The frame-start latch was
# `_g0 <= 0`. When the pose leaves the frustum the render returns nothing, |g|
# is EXACTLY zero, _g0 latches to zero - which still reads as "not started" -
# and three things go wrong at once:
#
#   1. every iteration counts as a frame start, so the barred denominator
#      reports iterations instead of frames (measured: 'barred 7/12387' over
#      248 frames, i.e. 12139 of 32247 iterations rendering nothing);
#   2. the progress gate `_g_best < _g0 * (1 - eps)` becomes `_g_best < 0`,
#      false forever, so the frame CANNOT stop and burns its whole budget
#      while momentum carries the pose further out;
#   3. gn == 0 passes the anomaly test trivially, so an empty render is
#      ACCEPTED into the healthy-frame reference and drags it toward zero.
#
# End state: GSLAM TUM fr1 at ATE 291.37cm, from a config whose only change
# from a 2.05cm arm was a longer iteration budget.
_e24 = torch.zeros(6, dtype=DT)
_e24[1] = 1.0

# -- (1) the denominator counts FRAMES, over a long frame --------------------
_pd = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
for _f in range(3):
    _pd.carry_frame(transport=None)
    for _i in range(40):
        _pd.step(_e24 * (1.0 - 0.02 * _i))
check("  the barred denominator counts frames, not iterations",
      float(_pd._judged_dev) == 3.0,
      f"judged = {float(_pd._judged_dev):.0f} over 3 frames x 40 iterations")

# -- (2) a dead frame is judged ONCE, not once per iteration -----------------
_pd.carry_frame(transport=None)
for _i in range(40):
    _pd.step(_e24 * 0.0)
check("  a frame with an empty render is still judged exactly once",
      float(_pd._judged_dev) == 4.0,
      f"judged = {float(_pd._judged_dev):.0f} - the old latch reported 43")
check("  and the empty iterations are counted and reported",
      float(_pd._empty_dev) == 40.0 and float(_pd._dead_dev) == 1.0,
      f"{float(_pd._empty_dev):.0f} empty iters over "
      f"{float(_pd._dead_dev):.0f} frame")
check("  summary() names it rather than hiding it in the denominator",
      "EMPTY RENDERS" in _pd.summary(),
      "the only visible symptom used to be a wrong denominator")

# -- (3) THE SPIRAL. A dead frame must be able to stop. ----------------------
check("  a dead frame reports a stall no patience can survive",
      float(_pd.stalled()) >= 1e8,
      f"stalled() = {float(_pd.stalled()):.3g} - it used to report 0 forever")

# The pre-fix behaviour, reproduced exactly so the claim is not just asserted:
# with |g_0| == 0 the progress gate can never open.
_gate = float(_pd._g_best) < float(_pd._g0) * (1.0 - _pd.stop_improve)
check("  and the progress gate alone would still hold it open",
      not _gate,
      "_g_best < _g0*(1-eps) is 0 < 0 - false however long the frame runs")

# -- (4) a dead frame must not poison the healthy-frame reference -----------
_pr = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01, dtype=DT)
for _f in range(6):
    _pr.carry_frame(transport=None)
    for _i in range(10):
        _pr.step(_e24 * (1.0 - 0.05 * _i))
_ref_before = float(_pr._g0_ema)
_pr.carry_frame(transport=None)
for _i in range(10):
    _pr.step(_e24 * 0.0)
check("  an empty render does NOT enter the healthy-frame reference",
      float(_pr._g0_ema) == _ref_before and _ref_before > 0.0,
      f"reference held at {_ref_before:.4f} through a dead frame")

# -- (5) and a healthy frame after one is unaffected ------------------------
_pr.carry_frame(transport=None)
_pr.step(_e24 * 1.0)
check("  the frame after recovers a normal relative gradient",
      abs(float(_pr.rel_grad()) - 1.0) < 1e-6 and float(_pr._dead) == 0.0,
      f"rel = {float(_pr.rel_grad()):.4f}, dead latch cleared by carry_frame")

# ---------------------------------------------------------------------------
print(chr(10) + "[25] the plateau anchor - a ratchet that stops too early")
# THE DEFECT THIS PINS. The counter compares against _g_best and then lowers
# _g_best to EVERY new minimum, including ones that did not beat the margin.
# So a run of sub-margin improvements never resets the counter AND slides the
# bar down with it: accumulated progress can never register, and a frame that
# is still converging is cut at `patience`.
#
# The standard definition (Keras EarlyStopping and every reference
# implementation) moves the anchor only on a significant improvement:
#     if current < best - min_delta: best = current; wait = 0
#     else:                          wait += 1
#
# A cut frame commits a half-converged pose, the map is built from it, and the
# next frame starts worse - the shape of the divergences observed at frames
# 230 and 300 rather than at the start of a run.
_e25 = torch.zeros(6, dtype=DT)
_e25[2] = 1.0
# 0.9% per iteration against a 1% margin: below the bar every single step,
# but 8.6% cumulatively over ten.
_curve = [0.991 ** _i for _i in range(14)]

_ratchet = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01,
                              stop_anchor="min", dtype=DT)
_ratchet.carry_frame(transport=None)
for _v in _curve:
    _ratchet.step(_e25 * _v)

_sig = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01,
                          stop_anchor="sig", dtype=DT)
_sig.carry_frame(transport=None)
for _v in _curve:
    _sig.step(_e25 * _v)

check("  the recorded anchor NEVER resets on a 0.9%/iter fall at a 1% margin",
      float(_ratchet.stalled()) >= 12,
      f"stall = {float(_ratchet.stalled()):.0f} after 14 iterations, "
      f"cumulative improvement {100 * (1 - _curve[-1]):.1f}%")
check("  and the standard anchor DOES reset - the frame keeps running",
      float(_sig.stalled()) < 12,
      f"stall = {float(_sig.stalled()):.0f}, anchor moves only when the "
      f"margin is actually beaten")
check("  the two disagree, which is the whole point of the flag",
      float(_ratchet.stalled()) != float(_sig.stalled()),
      f"{float(_ratchet.stalled()):.0f} vs {float(_sig.stalled()):.0f}")

# A GENUINE plateau must still stop under BOTH - the fix must not disable
# stopping, only stop cutting frames that are still converging.
for _name, _mode in (("min", "min"), ("sig", "sig")):
    _flat = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01,
                               stop_anchor=_mode, dtype=DT)
    _flat.carry_frame(transport=None)
    _flat.step(_e25 * 1.0)
    _flat.step(_e25 * 0.70)          # one real improvement, opens the gate
    for _ in range(14):
        _flat.step(_e25 * 0.70)      # then genuinely flat
    check(f"  a real plateau still stops under stop_anchor={_name}",
          float(_flat.stalled()) >= 12,
          f"stall = {float(_flat.stalled()):.0f} on a flat sequence")

# And the progress gate still protects a diverging frame under the new anchor.
_div = PosePreconditioner(dim=6, lr=0.004, stop_improve=0.01,
                          stop_anchor="sig", dtype=DT)
_div.carry_frame(transport=None)
for _v in [1.0 + 0.05 * _i for _i in range(14)]:
    _div.step(_e25 * _v)
check("  and a DIVERGING frame still reports 0 under the new anchor",
      float(_div.stalled()) == 0.0,
      "the progress gate is unaffected by where the anchor sits")

# ---------------------------------------------------------------------------
# [26] C0 - the two phases on the same axis
#
# The whole value of the measurement is that it is COMPARABLE to _step_sum. A
# 7-parameter (q, t) delta norm would have been cheaper, would have looked
# entirely plausible in a log, and would have answered a different question.
# ---------------------------------------------------------------------------
print("\n[26] C0 - the other phase's step, in the same units")

_worst_rt = 0.0
for _mag in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0):
    for _ in range(200):
        _xi = torch.randn(6, dtype=DT)
        _xi = _xi / _xi.norm() * _mag
        _rt = se3_log_capturable(se3_exp_capturable(_xi))
        _worst_rt = max(_worst_rt, float((_rt - _xi).norm() / _xi.norm()))
check("  se3_log_capturable inverts se3_exp_capturable over |xi| 1e-9..1",
      _worst_rt < 1e-9, f"worst relative round-trip {_worst_rt:.2e}")

# Pinned against the ORIGINAL, exactly as mat_to_quat_capturable is pinned
# against Shepperd. A branch-free rewrite that agrees with itself proves
# nothing; the claim is that it computes the same map as the validated one.
_worst_pin = 0.0
for _mag in (1e-6, 1e-4, 1e-2, 1e-1, 0.5):
    for _ in range(200):
        _xi = torch.randn(6, dtype=DT)
        _xi = _xi / _xi.norm() * _mag
        _T = se3_exp(_xi)
        _worst_pin = max(_worst_pin, float(
            (se3_log_capturable(_T) - se3_log(_T)).norm() / _mag))
check("  and agrees with gn_tracking.se3_log, the validated implementation",
      _worst_pin < 1e-9, f"worst relative disagreement {_worst_pin:.2e}")

# THE BAND THAT MATTERS. Accepted tracking steps measure ~1e-4, and the series
# boundary sits at 1e-2, so this is the range the number will actually be
# computed in.
_worst_use = 0.0
for _ in range(2000):
    _q = torch.randn(4, dtype=DT); _q = _q / _q.norm()
    _t = torch.randn(3, dtype=DT)
    _xi = torch.randn(6, dtype=DT); _xi = _xi / _xi.norm() * 1e-4
    _qn, _tn = apply_tangent_step(_q, _t, _xi, mat_to_quat_capturable)
    _worst_use = max(_worst_use, float(
        (tangent_of_pose_delta(_q, _t, _qn, _tn) - _xi).norm()))
check("  tangent_of_pose_delta recovers an applied step at the size actually used",
      _worst_use < 1e-11,
      f"worst ABSOLUTE error {_worst_use:.2e} against a 1e-4 step")

# THE GAUGE. cam_unnorm_rots is unnormalised and Adam moves its norm freely.
# That norm is an exact null direction of the loss, so if it registered as
# motion the measurement would credit the Adam phase with travel it did not do
# - and would do so ONLY for the phase that does not renormalise, which is
# precisely the phase under test.
_q = torch.randn(4, dtype=DT); _q = _q / _q.norm()
_t = torch.randn(3, dtype=DT)
# THE DTYPE TEST, AND IT IS THE ONE THAT NEARLY SHIPPED A GARBAGE NUMBER.
# Every other check in this group runs in float64; the tracker's pose params
# are float32, and this differences two ABSOLUTE poses to recover a ~1e-4
# RELATIVE motion. That is catastrophic cancellation twice (a near-identity
# R1 R0^T, and t1 - dR t0), and computed in float32 throughout it returns 98%
# relative error at |dxi|=1e-4 and 386% at 1e-5 - a plausible-looking log line
# that answers the direction/distance question wrongly.
#
# WHAT THE FLOOR ACTUALLY IS. The pose is STORED in float32, so the motion is
# only recoverable to the storage quantisation: ~6e-8 absolute on components of
# order 1, against a step of |dxi|. Measured, it lands at 2.8e-03 relative at
# |dxi|=1e-4 and 4.1e-02 at 1e-5, and no arithmetic can do better - the
# information is not in the tensor. What float64 buys is landing ON that floor
# (2.8e-03) instead of 350x above it (9.8e-01). The tolerances are the measured
# floor with headroom, not a preference.
#
# AND THE FLOOR IS FINE FOR WHAT C0 REPORTS, which is a MEAN over thousands of
# steps. Quantisation error is independent between iterations, so it averages
# down by sqrt(n): at ~5500 Adam steps in a full run the mean is good to ~4e-05
# relative. The per-step number is noisy; the statistic is not.
#
# Poses are moved in float64 and then CAST to float32, which is what the
# tracker's own storage does to Adam's update; comparing against a step that
# float32 apply_tangent_step never actually applied would measure that
# function's precision rather than this one's.
for _f32mag, _tol in ((1e-5, 6e-2), (1e-4, 5e-3), (1e-3, 6e-4)):
    _w32 = 0.0
    for _ in range(400):
        _q64 = torch.randn(4, dtype=DT); _q64 = _q64 / _q64.norm()
        _t64 = torch.randn(3, dtype=DT)
        _xi64 = torch.randn(6, dtype=DT); _xi64 = _xi64 / _xi64.norm() * _f32mag
        _qn64, _tn64 = apply_tangent_step(_q64, _t64, _xi64, mat_to_quat_capturable)
        # stored as the tracker stores them
        _a = (_q64.float(), _t64.float(), _qn64.float(), _tn64.float())
        _w32 = max(_w32, float(
            (tangent_of_pose_delta(*_a) - _xi64).norm() / _xi64.norm()))
    check(f"  float32-STORED poses (what the tracker has) at |dxi|={_f32mag:g}",
          _w32 < _tol,
          f"worst relative error {_w32:.2e} against a {_tol:g} quantisation "
          f"floor; float32 arithmetic throughout gives "
          f"{'9.8e-01' if _f32mag == 1e-4 else 'up to 3.9e+00'}")

check("  and pure gauge motion (q scaled) registers as ZERO travel",
      float(tangent_of_pose_delta(_q, _t, _q * 1.7, _t).norm()) < 1e-12,
      f"|dxi| = {float(tangent_of_pose_delta(_q, _t, _q * 1.7, _t).norm()):.1e} "
      f"for a 1.7x rescale that the loss cannot see")

# The counters are separate, and the Adam-phase one is on DEVICE - the host
# counter it would otherwise share is incremented in refactor(), which
# splatam.py skips during exactly this phase.
_c0 = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, dtype=DT)
_c0.carry_frame(transport=None)
_g0 = torch.tensor([1.0, 0, 0, 0, 0, 0], dtype=DT)
_c0.step(_g0); _c0.refactor()
_c0.note_applied_step(torch.tensor([3e-3, 0, 0, 0, 0, 0], dtype=DT))
_c0.note_applied_step(torch.tensor([5e-3, 0, 0, 0, 0, 0], dtype=DT))
check("  note_applied_step accumulates separately from the metric phase",
      float(_c0._astep_n) == 2.0 and _c0._step_n == 1,
      f"other-phase n = {float(_c0._astep_n):.0f} (device), "
      f"metric-phase n = {_c0._step_n} (host) - pooling them would describe "
      f"neither")
check("  mean and max are the other phase's, not a blend of both",
      abs(float(_c0._astep_sum) / 2 - 4e-3) < 1e-12
      and abs(float(_c0._astep_max) - 5e-3) < 1e-12,
      f"mean {float(_c0._astep_sum) / 2:.2e}, max {float(_c0._astep_max):.2e}")
check("  and summary() prints the ratio that answers direction-vs-distance",
      "ratio other/metric" in _c0.summary(),
      "computing it by hand from two log lines is how 0.02x-vs-0.21x happened")


# ---------------------------------------------------------------------------
# [27] C1 - the mid-frame restart, without the rule change
# ---------------------------------------------------------------------------
print("\n[27] C1 - PRE_RESTART: a momentum restart is not an optimiser switch")

def _run_frames(pre, n_frames, n_iters, g):
    """One frame at a time, in the loop's order: step() then refactor()."""
    for _ in range(n_frames):
        pre.carry_frame(transport=None)
        for _ in range(n_iters):
            pre.step(g)
            pre.refactor()

_G = torch.tensor([1.0, 0.5, -0.3, 0.2, 0.1, -0.4], dtype=DT)

# DEFAULT IS A BIT-EXACT NO-OP. Same precedent as PRE_MAX_STEP and STOP_ANCHOR:
# every number already measured must stay comparable, so the knob may not
# perturb the shipped path by a single ulp when it is off.
_off_a = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, dtype=DT)
_off_b = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, restart_at=0,
                            ramp_to=0, diag_after=0, dtype=DT)
_run_frames(_off_a, 2, 12, _G)
_run_frames(_off_b, 2, 12, _G)
check("  restart/ramp/diagonal-tail off leave the shipped path bit-identical",
      torch.equal(_off_a.m, _off_b.m) and torch.equal(_off_a.P, _off_b.P),
      "defaults unchanged, so every number already on record stays comparable")

# Fires exactly once per frame, at the right iteration.
_r = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, restart_at=5, dtype=DT)
_run_frames(_r, 3, 12, _G)
check("  the restart fires exactly once per frame",
      _r.restarts == 3, f"restarts = {_r.restarts} over 3 frames")

# THE OFF-BY-ONE. refactor() runs AFTER step(), so firing on _frame_iters ==
# restart_at makes iteration `restart_at` the FIRST to see a fresh moment -
# the same semantics the handoff has, where Adam's state is empty AT the
# handoff iteration and not one step later.
_r2 = PosePreconditioner(dim=6, lr=0.004, beta1=0.9, refactor_every=1,
                         restart_at=5, dtype=DT)
_r2.carry_frame(transport=None)
for _i in range(5):                     # iterations 0..4
    _r2.step(_G); _r2.refactor()
check("    m is zeroed BEFORE iteration restart_at, not after it",
      float(_r2.m.norm()) == 0.0 and _r2._t == 0,
      f"|m| = {float(_r2.m.norm()):.1e}, _t = {_r2._t} entering iteration 5")
_r2.step(_G)
check("    so the first post-restart step sees m_hat == the raw gradient",
      torch.allclose(_r2.m / (1 - 0.9 ** _r2._t), _G, atol=1e-12),
      "m = (1-b1)g and the bias clock is at t=1, so the correction is exact")

# THE BIAS CLOCK IS THE HALF THAT IS EASY TO FORGET. Zeroing m but not _t
# leaves the correction at 1/(1-beta1^t) for the OLD t, so the first step after
# the restart is short by (1-beta1)/(1-beta1^t) - and the arm would be
# measuring a stall rather than a restart.
#
# The shortfall grows with the restart point, toward (1-beta1) = 0.1: it is
# 0.21x at the restart_at=5 used here and 0.11x at a realistic restart_at=20.
# Both are large enough to change what the arm measures.
_shortfall = (1 - 0.9) / (1 - 0.9 ** 6)
_at20 = (1 - 0.9) / (1 - 0.9 ** 21)
check("    and zeroing m WITHOUT the clock would start 4.7x short here",
      abs(float((_r2.m / (1 - 0.9 ** _r2._t)).norm() / _G.norm()) - 1.0) < 1e-9
      and _shortfall < 0.25,
      f"correct = 1.00x the gradient; the unclocked version would be "
      f"{_shortfall:.2f}x at restart_at=5 and {_at20:.2f}x at restart_at=20")

# M AND P SURVIVE. The carry argument is that M is a property of the scene and
# m of the trajectory; a restart is a statement about the trajectory only.
_Mb = _r2.M.clone()
_r2.refactor()
check("  M and P are NOT reset - this isolates the moment, not the metric",
      float(_Mb.norm()) > 0 and float(_r2.M.norm()) > 0,
      "zeroing M as well would be a different experiment, and m0_iso is it")

# REQUESTED vs HAPPENED. A run whose frames all stop before restart_at is
# bit-identical to the control while carrying this arm's tag.
_never = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, restart_at=30,
                            dtype=DT)
_run_frames(_never, 4, 12, _G)
check("  a frame that stops before restart_at never fires it, and says so",
      _never.restarts == 0 and "NEVER FIRED" in _never.summary(),
      "otherwise the arm reports as a restart while being the control")


# ---------------------------------------------------------------------------
# [28] C2 - the continuous handoff
# ---------------------------------------------------------------------------
print("\n[28] C2 - PRE_RAMP: the handoff's shape, with one optimiser")

_ref = 0.004 * math.sqrt(6)

# w = 0 reproduces the un-ramped step exactly; w = 1 is normalised gradient
# descent at Adam's magnitude - the SAME expression the no-metric branch uses.
_plain = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, dtype=DT)
_hard = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                           ramp_from=4, ramp_to=4, dtype=DT)
_plain.carry_frame(transport=None); _hard.carry_frame(transport=None)
_d_plain = _d_hard = None
for _i in range(4):
    _d_plain = _plain.step(_G); _plain.refactor()
    _d_hard = _hard.step(_G); _hard.refactor()
check("  before ramp_from the step is bit-identical to the un-ramped one",
      torch.equal(_d_plain, _d_hard), "the ramp is inert until it engages")

_d_hard = _hard.step(_G)
_d_plain = _plain.step(_G)
check("  at w=1 the step is normalised gradient descent at Adam's magnitude",
      abs(float(_d_hard.norm()) - _ref) < 1e-12,
      f"|d| = {float(_d_hard.norm()):.4e} against ref = lr*sqrt(6) = {_ref:.4e}")
_mh = _hard.m / (1 - 0.9 ** _hard._t)
check("    and its DIRECTION is the raw first moment, not the metric's",
      float(torch.nn.functional.cosine_similarity(
          _d_hard, -_mh, dim=0)) > 1 - 1e-12,
      f"cos(d, -m_hat) = {float(torch.nn.functional.cosine_similarity(_d_hard, -_mh, dim=0)):.12f}")
check("    which is a DIFFERENT step from the metric's, so the knob does something",
      float((_d_hard - _d_plain).norm() / _d_plain.norm()) > 1e-3,
      f"relative difference {float((_d_hard - _d_plain).norm() / _d_plain.norm()):.3f}")

# The schedule itself.
_lin = PosePreconditioner(dim=6, lr=0.004, ramp_from=10, ramp_to=20, dtype=DT)
check("  the linear schedule is 0 before, 0.5 at the midpoint, 1 after",
      (_lin._ramp_weight(9) == 0.0 and _lin._ramp_weight(15) == 0.5
       and _lin._ramp_weight(20) == 1.0 and _lin._ramp_weight(40) == 1.0),
      "w(9)=0, w(15)=0.5, w(20)=1, w(40)=1")
check("    and from == to is a hard switch - the handoff's own shape",
      _hard._ramp_weight(3) == 0.0 and _hard._ramp_weight(4) == 1.0,
      "so the arm can be run as a drop-in replacement for the phase split")

# PER FRAME, like the handoff it replaces. Without the reset in carry_frame one
# frame would ramp and every later frame would start already at w=1 - a
# different method entirely, wearing this arm's tag.
_hard.carry_frame(transport=None)
check("  the ramp resets at every frame boundary",
      float(_hard._ramp_w) == 0.0,
      "iteration 20 of every frame, not iteration 20 of the run")

# P stays the pure metric. Writing the blend into P would corrupt the cached
# eigendecomposition refactor() maintains, and the corruption would be
# invisible - P would simply be a worse preconditioner every replay.
_pre_P = _hard.P.clone()
_hard.step(_G)
check("  the blend does not touch P, so refactor()'s cache stays honest",
      torch.equal(_hard.P, _pre_P),
      "the lerp is over STEPS; P remains the metric it says it is")

# A backwards ramp is a configuration error, not something to interpret later.
try:
    PosePreconditioner(dim=6, lr=0.004, ramp_from=30, ramp_to=10, dtype=DT)
    _raised = False
except ValueError:
    _raised = True
check("  ramp_to < ramp_from is refused at construction",
      _raised, "a backwards ramp would run the metric phase LAST")

# ---------------------------------------------------------------------------
# [29] the pre-cap step profile
#
# THE POINT OF PRE-CAP. Post-cap, every excursion reads as exactly the cap and
# the distribution that produced it is gone - which is why 'max |d| = 2.45e-01
# to three digits across configs' could be identified as a constant and never
# explained. This is the instrument that would have explained it.
# ---------------------------------------------------------------------------
print("\n[29] the pre-cap step profile")

# A PLANTED DISTRIBUTION, so the answer is known rather than inspected. Drive
# step() with gradients chosen to land at requested multiples of ref, then ask
# the profile to report them back.
_pp = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, max_step_mult=10.0,
                         dtype=DT)
_pp.carry_frame(transport=None)
_ref = 0.004 * math.sqrt(6)
_e29 = torch.zeros(6, dtype=DT); _e29[0] = 1.0
for _ in range(40):
    _pp.step(_e29); _pp.refactor()
check("  a profile is produced once steps have run",
      _pp.step_profile() != "",
      "empty until the first capped step, so a no-metric run prints nothing")
# EXACTLY ONE STEP IS MISSING, AND IT SHOULD BE. The run's first step goes
# through the no-metric branch, which takes a normalised step and is not
# subject to the trust region - so there is no pre-cap norm to record. Pinning
# the count at steps-1 rather than steps means a profile that silently stopped
# recording (a frozen counter, a rebound buffer) cannot pass.
check("  every capped step is counted exactly once",
      float(_pp._raw_hist.sum()) == 39.0,
      f"histogram total {float(_pp._raw_hist.sum()):.0f} = 40 steps - 1, the "
      f"run's first step running before any metric exists")

# THE CAP IS RECORDED AS REQUESTED-vs-HAPPENED. An arm that reports
# max_step_mult=10 while never reaching 2 is not being bounded by its cap, and
# nothing in the summary used to say so.
_tight = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                            max_step_mult=0.01, dtype=DT)
_tight.carry_frame(transport=None)
for _ in range(20):
    _tight.step(_e29); _tight.refactor()
_loose = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                            max_step_mult=1e6, dtype=DT)
_loose.carry_frame(transport=None)
for _ in range(20):
    _loose.step(_e29); _loose.refactor()
check("  a cap that binds is reported as binding",
      float(_tight._clip_n) > 0, f"clipped {int(_tight._clip_n)} steps at 0.01x")
check("    and one that never binds reports zero",
      float(_loose._clip_n) == 0.0,
      "so 'max_step_mult=10' can no longer hide never having been reached")

# THE PROFILE IS PRE-CAP, WHICH IS THE WHOLE CLAIM. Two runs differing ONLY in
# the cap must record the SAME distribution - if the histogram moved, it is
# measuring the clip rather than the step that provoked it.
check("  the recorded distribution does NOT depend on the cap",
      torch.allclose(_tight._raw_hist, _loose._raw_hist),
      "same steps, caps 8 orders of magnitude apart, identical histogram - "
      "so one run evaluates every candidate cap offline")

# The would-clip tail is a real cumulative share, monotone by construction.
_rows = []
for _tok in _loose.step_profile().splitlines()[1].split()[3:]:
    _lo, _pc = _tok.split(":")
    _rows.append((float(_lo.rstrip("x")), float(_pc.rstrip("%"))))
check("  the would-clip tail is monotone decreasing in the cap",
      all(_rows[_i][1] >= _rows[_i + 1][1] for _i in range(len(_rows) - 1)),
      "a looser cap cannot clip more than a tighter one: "
      + " ".join(f"{a:g}x:{b:.0f}%" for a, b in _rows))

# THE BANDS SPLIT BY WITHIN-FRAME ITERATION, and _fi_dev has to advance under
# what would be a replay. It is maintained in refactor() for exactly the reason
# _step_n is; if it stood still every step would land in band 0 and the
# early-vs-late question - the one the bands exist to answer - would silently
# read 'all early'.
_bands = PosePreconditioner(dim=6, lr=0.004, refactor_every=1, dtype=DT)
_bands.carry_frame(transport=None)
for _ in range(45):
    _bands.step(_e29); _bands.refactor()
check("  steps are distributed across iteration bands, not piled into band 0",
      int((_bands._band_n > 0).sum()) == 5,
      f"band counts {[int(x) for x in _bands._band_n]} for a 45-iteration frame "
      f"- all in band 0 is the signature of a frozen counter")
# THE EDGE CASE THAT WAS WRONG. An iteration sitting exactly ON a band edge
# must land in the band that NAMES it: iteration 5 is it5-9, not it0-4.
# torch.bucketize defaults to the other convention, which shifted every band by
# its own first element - and it would have read as a plausible profile.
# 44, not 45: the run's first step has no metric and is not capped.
check("    and the band counts match the edges the names claim",
      [int(x) for x in _bands._band_n] == [4, 5, 10, 20, 5],
      f"{[int(x) for x in _bands._band_n]} for it0-4/it5-9/it10-19/it20-39/it40+ "
      f"over 45 iterations (band 0 is 4, not 5, because iteration 0 is "
      f"uncapped); bucketize's default gives [5,5,10,20,4]")

# PER FRAME, not per run - the bands describe position WITHIN a frame.
_bands.carry_frame(transport=None)
_bands.step(_e29)
check("  the band counter resets at a frame boundary",
      int(_bands._band_n[0]) == 5,
      "band 0 goes 4 -> 5: iteration 0 of frame 2 is band 0, not band 4. "
      "Without the reset in carry_frame the bands would describe position in "
      "the RUN, and every frame after the first would read as all-late")

# ---------------------------------------------------------------------------
# [30] the restart's effect on M - scale without shape
#
# MEASURED MOTIVATION. At it20-39 the metric steps at 0.10x ref where the Adam
# tail steps at 0.33x. Clearing m alone recovers 39% of that, matching the EMA
# horizons another 17%; half the gap is M still carrying the acquisition
# phase's large gradients. On a decaying gradient trace(M) ends a frame 34x
# above |g|^2, and P ~ M^-1/2 turns that into a ~5.8x step suppression.
# ---------------------------------------------------------------------------
print("\n[30] the restart's effect on M")

def _decay_frame(pre, n=30, base=None):
    """One frame with a geometrically decaying gradient - the real regime."""
    base = _G if base is None else base
    pre.carry_frame(transport=None)
    g = None
    for i in range(n):
        g = base * (0.9 ** i)
        pre.step(g)
        pre.refactor()
    return g

# DEFAULT IS OFF, and off must be bit-identical to the m-only arm that produced
# the 0.19x measurement - otherwise that number stops being reproducible.
_r_off = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, restart_at=20,
                            dtype=DT)
_r_off2 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, restart_at=20,
                             restart_m="off", dtype=DT)
_decay_frame(_r_off); _decay_frame(_r_off2)
check("  restart_m defaults to off, bit-identical to the m-only restart",
      torch.equal(_r_off.M, _r_off2.M) and torch.equal(_r_off.m, _r_off2.m),
      "the 0.19x measurement stays reproducible")

# THE STALENESS ITSELF, which is the thing being corrected.
_stale = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dtype=DT)
_gl = _decay_frame(_stale)
_ratio = float(torch.diagonal(_stale.M).sum()) / float(_gl.norm() ** 2)
check("  without a restart, M ends a decaying frame far above |g|^2",
      _ratio > 10.0,
      f"trace(M)/|g|^2 = {_ratio:.1f}x, so P ~ M^-1/2 suppresses the step by "
      f"~{math.sqrt(_ratio):.1f}x exactly when the frame has stopped converging")

# TRACE MODE: scale re-anchored to the current gradient, exactly.
_r_tr = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, restart_at=5,
                           restart_m="trace", dtype=DT)
_r_tr.carry_frame(transport=None)
_M_before = None
for _i in range(5):
    _g5 = _G * (0.9 ** _i)
    _r_tr.step(_g5)
    if _i == 4:
        _M_before = _r_tr.M.clone()
    _r_tr.refactor()
check("  trace mode re-anchors trace(M) to |g|^2 exactly",
      abs(float(torch.diagonal(_r_tr.M).sum()) / float(_g5.norm() ** 2) - 1.0) < 1e-10,
      f"trace(M)/|g|^2 = "
      f"{float(torch.diagonal(_r_tr.M).sum()) / float(_g5.norm()**2):.10f}")

# AND THE SHAPE SURVIVES, which is the entire design. A scalar multiply leaves
# the eigenvectors untouched, so the carried coupling - the method's central
# claim - is not thrown away to fix a magnitude.
# Compared as NORMALISED matrices, not via eigh. A rescale is a scalar
# multiple, so M/|M| is invariant under it - exact, and free of the ambiguity
# eigh has on a matrix with degenerate eigenvalues, where the returned basis
# for a null space is arbitrary and two calls need not agree.
_nb = _M_before / _M_before.norm()
_na = _r_tr.M / _r_tr.M.norm()
check("    while leaving M's SHAPE untouched - a scalar multiple, nothing more",
      float((_nb - _na).abs().max()) < 1e-12,
      f"max |M/|M| - M'/|M'|| = {float((_nb - _na).abs().max()):.1e}; the "
      f"coupling the carry exists to accumulate survives the restart, and only "
      f"the magnitude - stale by construction - is re-anchored")

# ISO MODE: the contrast that can falsify the coupling claim at this point.
_r_iso = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, restart_at=5,
                            restart_m="iso", dtype=DT)
_r_iso.carry_frame(transport=None)
for _i in range(5):
    _g5 = _G * (0.9 ** _i)
    _r_iso.step(_g5); _r_iso.refactor()
_off_diag = _r_iso.M - torch.diag(torch.diagonal(_r_iso.M))
check("  iso mode discards the shape as well, as the contrast arm",
      float(_off_diag.abs().max()) < 1e-12
      and abs(float(torch.diagonal(_r_iso.M).sum()) / float(_g5.norm()**2) - 1.0) < 1e-10,
      "same trace as trace-mode, no coupling - if the two score the same, the "
      "carried coupling was contributing nothing at the restart point")

# P MUST BE REBUILT AT THE RESTART. Rescaling M without forcing a refactor
# would leave the step running on the OLD P for up to refactor_every
# iterations - i.e. for most of the tail this exists to fix, so the arm would
# measure almost nothing and look like a null result.
check("  the rescale forces P to be rebuilt immediately",
      _r_tr._since_refactor == 0,
      "otherwise the new scale would not reach the step for up to "
      "refactor_every iterations - the tail this is meant to fix")

try:
    PosePreconditioner(dim=6, lr=0.004, restart_m="sometimes", dtype=DT)
    _bad_m = False
except ValueError:
    _bad_m = True
check("  an unrecognised restart_m is refused at construction",
      _bad_m, "a typo must not silently select the shipped behaviour")

# ---------------------------------------------------------------------------
# [31] the drift guard - a degradation the |g_0| guard cannot see
#
# MEASURED MOTIVATION. On the diverging full-length run the mean step per band
# quadrupled over ~50 frames while `barred` sat at 9-11%. The existing guard
# reads |g_0|; this one reads |d|/ref. Test one below is why that matters: the
# step ratio is EXACTLY scale-free in |g|, so the two observables can move
# independently and a |g_0| guard is blind to a coherence change by
# construction.
# ---------------------------------------------------------------------------
print()
print("[31] the drift guard - a degradation the |g_0| guard cannot see")

def _run_frame(pre, g, iters=30):
    pre.carry_frame(transport=None)
    for _ in range(iters):
        pre.step(g)
        pre.refactor()

# WHY THE GUARD CANNOT WATCH |g|. M ~ E[gg^T] scales as g^2 and m as g, so
# M^{-1/2} m is invariant to gradient magnitude. Scaling |g| by 4 changes the
# step ratio by nothing at all - which is the method's defining property, and
# also the reason a |g_0|-based guard and a step-based one are measuring
# genuinely different things.
_ratios = []
for _mag in (1.0, 2.0, 4.0):
    _p = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dtype=DT)
    _run_frame(_p, _G * _mag)
    _ratios.append(float(_p._step_sum) / _p._step_n / (0.004 * math.sqrt(6)))
check("  the step ratio is EXACTLY scale-free in |g|",
      max(_ratios) - min(_ratios) < 1e-12,
      f"|d|/ref = {_ratios[0]:.4f} at 1x, 2x and 4x the gradient - so |g_0| and "
      f"|d|/ref can move independently, and a |g_0| guard cannot see a "
      f"coherence change")

# OFF BY DEFAULT, bit-exact when off.
_d_off = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dtype=DT)
_d_off2 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, drift_mult=0.0,
                             dtype=DT)
for _ in range(5):
    _run_frame(_d_off, _G); _run_frame(_d_off2, _G)
check("  drift_mult=0 leaves the shipped path bit-identical",
      torch.equal(_d_off.M, _d_off2.M) and float(_d_off.drifted()) == 0.0,
      "inert when off, and drifted() returns 0")

# INERT UNTIL THE BASELINE EXISTS - a guard that fired while learning would bar
# the very frames it is calibrating against.
_d = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, drift_mult=2.0,
                        drift_warmup=20, dtype=DT)
for _ in range(5):
    _run_frame(_d, _G)
check("  it stays inert before the baseline is frozen",
      float(_d._dref) == 0.0 and float(_d.drifted()) == 0.0,
      "nothing to judge against yet")

for _ in range(16):
    _run_frame(_d, _G)
check("    then freezes a baseline LEARNED from the scene",
      float(_d._dref) > 0.0,
      f"baseline {float(_d._dref):.3f}x ref from {_d.drift_warmup} frames - the "
      f"only constant the guard adds is the dimensionless multiplier")
_base_ref = float(_d._dref)

# THE DEGRADATION. Planted directly on the observable rather than simulated
# through the physics: what raises the step ratio in a real run is a change in
# gradient COHERENCE, and reproducing that faithfully would test the renderer,
# not the guard. Each frame sits 3x the baseline.
for _ in range(12):
    _run_frame(_d, _G)
    _d._dsum.copy_(_d._dn * (_base_ref * 3.0))
    check_flag = float(_d.drifted())
_run_frame(_d, _G)          # roll up the last planted frame
check("  frames above the baseline are flagged",
      float(_d._drift_n) >= 12,
      f"drift-barred {int(float(_d._drift_n))} frames at 3x baseline")
check("    and the baseline does not move while they are flagged",
      abs(float(_d._dref) - _base_ref) < 1e-12,
      f"frozen at {float(_d._dref):.3f}x ref through 12 elevated frames - "
      f"cumulative drift is removed by construction, not bounded per frame")
check("    the summary reports it",
      "drift-barred" in _d.summary(),
      "requested-vs-happened: a guard that never fires must say so")

# A STEADY SCENE MUST NOT BE FLAGGED, or the guard buys stability by spending
# the whole iteration budget on every frame.
_q = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, drift_mult=2.0,
                        drift_warmup=20, dtype=DT)
for _ in range(40):
    _run_frame(_q, _G)
check("  a steady sequence is never flagged",
      float(_q._drift_n) == 0.0,
      f"drift-barred 0/{40 - _q.drift_warmup} on a flat run - a guard that "
      f"fires on everything is a slower tracker, not a safer one")

# ---------------------------------------------------------------------------
# [32] an empty render must not teach the metric anything
#
# THE BUG. When the pose leaves the frustum the render returns nothing and |g|
# is EXACTLY zero, but the EMA still ran: M *= beta2 every iteration with
# nothing added. M collapses geometrically and P = M^{-1/2} grows as
# beta2^{-n/2}. Measured on a diverging full-length run with 923 empty
# iterations: a requested step of 9859x ref, it0-4 band averaging 152x.
#
# That is the positive feedback that makes a brief excursion permanent - the
# first gradient that DOES come back is multiplied by a metric orders of
# magnitude too large. Companion to 3d381d1, which fixed the same event for the
# stopping logic and left the metric unguarded.
# ---------------------------------------------------------------------------
print()
print("[32] an empty render must not teach the metric anything")

_e32 = torch.tensor([1.0, 0.4, -0.2, 0.3, -0.1, 0.2], dtype=DT)
_zero = torch.zeros(6, dtype=DT)

def _with_deaths(freeze, n_dead):
    pre = PosePreconditioner(dim=6, lr=0.004, refactor_every=10,
                             dead_freeze=freeze, dtype=DT)
    pre.carry_frame(transport=None)
    for _ in range(30):                     # healthy
        pre.step(_e32); pre.refactor()
    _M_live = pre.M.clone()
    for _ in range(n_dead):                 # frustum exit
        pre.step(_zero); pre.refactor()
    # MEASURED BEFORE THE RECOVERY STEP. That step adds (1-beta2) g g^T, which
    # dwarfs a collapsed M and restores the trace - measuring after it hides
    # the entire effect (it read 6.4e-02 instead of 2.1e-07).
    _M_dead = pre.M.clone()
    _d_back = pre.step(_e32)                # the render comes back
    return pre, _M_live, _M_dead, _d_back

_ref32 = 0.004 * math.sqrt(6)

# THE OLD BEHAVIOUR, reproduced so the fix has something to be measured against.
_old, _Mo, _Md, _do = _with_deaths(False, 300)
_collapse = float(torch.diagonal(_Md).sum()) / float(torch.diagonal(_Mo).sum())
check("  unguarded, 300 empty iterations collapse M toward zero",
      abs(_collapse / (0.95 ** 300) - 1.0) < 1e-6,
      f"trace(M) fell to {_collapse:.2e} of its live value, against "
      f"beta2^300 = {0.95**300:.2e} - the decay is exactly the EMA running on "
      f"nothing, which is the whole mechanism")
check("    and the first recovered gradient pins the trust region",
      float(_do.norm()) / _ref32 > 9.9,
      f"requested {float(_do.norm())/_ref32:.1f}x ref on the frame's return, "
      f"i.e. exactly max_step_mult - the same 'max |d| sits on the cap to "
      f"three digits' signature the record could never explain. The cap bounds "
      f"the magnitude but not the DIRECTION, which came from a metric built "
      f"out of nothing")

# THE FIX.
_new, _Mn, _Mnd, _dn = _with_deaths(True, 300)
check("  frozen, M is untouched by the empty iterations",
      torch.equal(_Mnd, _Mn),
      "bit-identical across 300 empty renders - there is nothing to learn "
      "from a render that returned nothing")
check("    and the recovered step is sane",
      float(_dn.norm()) / _ref32 < 2.0,
      f"requested {float(_dn.norm())/_ref32:.2f}x ref against the unguarded "
      f"{float(_do.norm())/_ref32:.0f}x")

# NO STEP AT ALL while dead. Freezing m alone would leave it holding the last
# live gradient, so the optimiser would keep pushing the pose in a stale
# direction - under the OLD behaviour m decayed to zero and the step died with
# it, so freezing without suppressing would be strictly worse.
_p32 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dtype=DT)
_p32.carry_frame(transport=None)
for _ in range(30):
    _p32.step(_e32); _p32.refactor()
_dead_steps = [float(_p32.step(_zero).norm()) for _ in range(20)]
check("  and no step is taken while the render is empty",
      max(_dead_steps) == 0.0,
      f"max |d| = {max(_dead_steps):.1e} over 20 empty iterations - nothing to "
      f"learn and nowhere to go")
check("    which the summary reports",
      "froze m/M on" in _p32.summary(),
      "requested-vs-happened for a guard that only fires on broken runs")

# THE CLAIM THAT LETS THIS DEFAULT TO ON. gn > 0 on every iteration that
# rendered anything, so on a healthy run the guard is not merely harmless - it
# is bit-exact. If that ever stops being true, this arm silently changes every
# number in both ladders.
_h1 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dead_freeze=True,
                         dtype=DT)
_h2 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10, dead_freeze=False,
                         dtype=DT)
for _p, _acc in ((_h1, []), (_h2, [])):
    _p.carry_frame(transport=None)
    for _i in range(40):
        _acc.append(_p.step(_e32 * (0.97 ** _i)))
        _p.refactor()
    _p._acc = _acc
check("  on a healthy run the guard is BIT-EXACT, not merely harmless",
      torch.equal(_h1.M, _h2.M) and torch.equal(_h1.m, _h2.m)
      and all(torch.equal(a, b) for a, b in zip(_h1._acc, _h2._acc)),
      "identical M, m and every step over 40 iterations - which is why this is "
      "on by default instead of behind a measurement flag")


# ---------------------------------------------------------------------------
# [33] full-matrix acquisition -> diagonal second-moment refinement
#
# This is the no-handoff rule supported by the ramp experiment: r40 retained
# all 20 keyframes through the frame-347 stress point and held ATE at 4.23 cm,
# but its normalised-gradient tail sat exactly at 1x ref on every step. The
# diagonal tail keeps Adam's learned per-coordinate scale from the SAME M,
# without another optimiser, state transfer, or constant-norm fallback.
# ---------------------------------------------------------------------------
print()
print("[33] full-matrix acquisition -> diagonal refinement")

_full33 = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                             max_step_mult=100.0, dtype=DT)
_diag33 = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                             max_step_mult=100.0, diag_after=20,
                             trace_steps=True, dtype=DT)
_full33.carry_frame(transport=None)
_diag33.carry_frame(transport=None)
_same_before = True
for _i in range(20):
    _g33 = torch.tensor([1.0 + .03 * _i, -.4, .2 + .01 * _i,
                         .7, -1.1 + .02 * _i, .35], dtype=DT)
    _df = _full33.step(_g33)
    _dd = _diag33.step(_g33)
    _same_before &= torch.equal(_df, _dd)
    _M_before_switch = _diag33.M.clone()
    _m_before_switch = _diag33.m.clone()
    _full33.refactor(); _diag33.refactor()

check("  iterations 0-19 are bit-identical to the full-matrix control",
      _same_before,
      "the new arm changes no acquisition step")
check("  switching factors does not reset or rewrite either moment",
      torch.equal(_diag33.M, _M_before_switch)
      and torch.equal(_diag33.m, _m_before_switch),
      "m and the full 6x6 M remain continuous across the switch")
check("  the cached tail factor is diagonal while the acquisition factor stays full",
      torch.count_nonzero(_diag33.P_diag - torch.diag(torch.diagonal(
          _diag33.P_diag))) == 0
      and torch.count_nonzero(_diag33.P - torch.diag(torch.diagonal(
          _diag33.P))) > 0,
      "P_diag changes the applied factor; P and M retain their coupling")
_M_eff33 = (_diag33.M / (1.0 - _diag33.beta2 ** _diag33._frame_iters)
            if _diag33._M_cold else _diag33.M)
check("    and its elementwise cache exactly matches shrink=0",
      torch.allclose(_diag33.P_diag, _inv_sqrt_damped(
          _M_eff33, _diag33.tau, _diag33.eps, 0.0),
          atol=1e-12, rtol=1e-12),
      "the faster cache changes cost, not diagonal-tail arithmetic")

_g33 = torch.tensor([.6, -.8, .1, .5, -.3, .9], dtype=DT)
_d_diag = _diag33.step(_g33)
_d_full = _full33.step(_g33)
_mhat33 = _diag33.m / (1.0 - _diag33.beta1 ** _diag33._t)
_want33 = -_diag33.lr * (_diag33.P_diag @ _mhat33)
check("  iteration 20 uses diag(M)^-1/2 from the same second moment",
      torch.allclose(_d_diag, _want33, atol=1e-12, rtol=1e-12),
      f"max|d-d_diag|={float((_d_diag - _want33).abs().max()):.2e}")
_snap33 = _diag33.diagnostic_snapshot()
_snapP33 = torch.tensor(_snap33["P"], dtype=DT)
check("    and forensic traces report the effective diagonal factor",
      torch.count_nonzero(_snapP33 - torch.diag(torch.diagonal(_snapP33))) == 0,
      "a failing tail trace must not be mislabeled with the cached full P")
check("    and it is genuinely different from the full-matrix direction",
      float((_d_diag - _d_full).norm() / _d_full.norm()) > 1e-3,
      f"relative step difference {float((_d_diag - _d_full).norm() / _d_full.norm()):.3f}")

# Production uses PRE_RESTART=20. Both events must land on the same exact
# iteration: step 20 sees a fresh first moment through the diagonal factor.
_both33 = PosePreconditioner(dim=6, lr=0.004, refactor_every=10,
                             restart_at=20, diag_after=20, dtype=DT)
_both33.carry_frame(transport=None)
for _ in range(20):
    _both33.step(_g33); _both33.refactor()
check("  restart and diagonal transition fire together at iteration 20",
      _both33.restarts == 1 and float(_both33._diag_w) == 1.0
      and torch.count_nonzero(_both33.m) == 0 and _both33._t == 0,
      "step 20 receives a fresh m and the diagonal factor")

# Every frame starts with the full cache again. The tail must not leak across
# the frame boundary simply because P_diag remains allocated.
_full33.refactor(); _diag33.refactor()
_full33.carry_frame(transport=None); _diag33.carry_frame(transport=None)
_first_full = _full33.step(_g33)
_first_diag_arm = _diag33.step(_g33)
check("  a new frame returns to full-matrix acquisition bit-exactly",
      float(_diag33._diag_w) == 0.0 and torch.equal(_first_full, _first_diag_arm),
      "the previous frame's diagonal tail cannot leak into iteration 0")
check("  the summary reports that the diagonal tail actually engaged",
      "diagonal tail @it20" in _diag33.summary()
      and "NEVER ENGAGED" not in _diag33.summary(),
      "requested-vs-happened is visible in every run log")

_never33 = PosePreconditioner(dim=6, lr=0.004, refactor_every=1,
                              diag_after=30, dtype=DT)
_run_frames(_never33, 3, 12, _G)
check("  frames stopping before the switch are identified as the control",
      "NEVER ENGAGED" in _never33.summary(),
      "a tagged arm cannot silently execute zero diagonal steps")

for _kwargs33 in ({"diag_after": 20, "ramp_from": 20, "ramp_to": 20},
                  {"diag_after": 20, "bfgs": True},
                  {"diag_after": -1}):
    try:
        PosePreconditioner(dim=6, lr=0.004, dtype=DT, **_kwargs33)
        _bad33 = False
    except ValueError:
        _bad33 = True
    check("  incompatible or backwards diagonal-tail configuration is refused",
          _bad33, str(_kwargs33))

# ---------------------------------------------------------------------------
# 34. AUTO-SCALED FULL ACQUISITION WITH PER-COORDINATE DIAGONAL REFINEMENT
# ---------------------------------------------------------------------------
# MonoGS's known-good acquisition phase uses target_step_norm, derived from
# three rotation rates and three translation rates. Keeping that normalisation
# after the diagonal switch pins every tail proposal to exactly 1x ref; the
# 100-frame smoke run exposed precisely that signature. The tail must recover
# the original Adam rate vector while the first 20 steps remain bit-exact.
print("\n[34] auto-scaled acquisition -> per-coordinate diagonal tail")
_rates34 = torch.tensor([.003, .003, .003, .001, .001, .001], dtype=DT)
_target34 = float(_rates34.norm())
_control34 = PosePreconditioner(
    dim=6, lr=.001, target_step_norm=_target34, refactor_every=10, dtype=DT)
_legacy34 = PosePreconditioner(
    dim=6, lr=.001, target_step_norm=_target34, refactor_every=10,
    diag_after=20, dtype=DT)
_tail34 = PosePreconditioner(
    dim=6, lr=.001, target_step_norm=_target34, refactor_every=10,
    diag_after=20, diag_lr=_rates34, dtype=DT)
for _p34 in (_control34, _legacy34, _tail34):
    _p34.carry_frame(transport=None)

_same34 = True
for _i in range(20):
    _g34 = torch.tensor([1.0 + .02 * _i, -.5, .2 + .01 * _i,
                         .8, -1.0 + .01 * _i, .3], dtype=DT)
    _dc34 = _control34.step(_g34)
    _dl34 = _legacy34.step(_g34)
    _dt34 = _tail34.step(_g34)
    _same34 &= torch.equal(_dc34, _dl34) and torch.equal(_dc34, _dt34)
    _control34.refactor(); _legacy34.refactor(); _tail34.refactor()
check("  custom tail rates leave iterations 0-19 bit-identical",
      _same34,
      "the established MonoGS full-matrix acquisition phase is unchanged")

_g34 = torch.tensor([.7, -.9, .15, .45, -.25, .85], dtype=DT)
_dlegacy34 = _legacy34.step(_g34)
_dtail34 = _tail34.step(_g34)
_mhat34 = _tail34.m / (1.0 - _tail34.beta1 ** _tail34._t)
_raw34 = -_rates34 * (_tail34.P_diag @ _mhat34)
_cap34 = _target34 * _tail34.max_step_mult
_want34 = _raw34 * min(1.0, _cap34 / max(float(_raw34.norm()), 1e-30))
check("  iteration 20 uses MonoGS's coordinate-wise Adam rates",
      torch.allclose(_dtail34, _want34, atol=1e-12, rtol=1e-12),
      f"max|d-d_adamlr|={float((_dtail34 - _want34).abs().max()):.2e}")
check("    rather than pinning the tail to target_step_norm",
      not torch.isclose(_dtail34.norm(), torch.tensor(_target34, dtype=DT),
                        atol=1e-8, rtol=1e-8)
      and torch.isclose(_dlegacy34.norm(), torch.tensor(_target34, dtype=DT),
                        atol=1e-12, rtol=1e-12),
      f"custom {float(_dtail34.norm()):.4e}, legacy {float(_dlegacy34.norm()):.4e}, "
      f"target {_target34:.4e}")
check("    and the run summary exposes the effective tail-rate range",
      "lr 0.001..0.003" in _tail34.summary(),
      "a runtime wiring failure must be visible in the log")
check("    target-scaled profiles label their reference honestly",
      "ref=target_step_norm" in _tail34.step_profile(),
      "MonoGS does not use scalar lr*sqrt(n) as its reference")

for _kwargs34 in ({"diag_lr": _rates34},
                  {"diag_after": 20, "diag_lr": [1e-3] * 5},
                  {"diag_after": 20, "diag_lr": [1e-3] * 5 + [-1e-3]}):
    try:
        PosePreconditioner(dim=6, lr=.001, dtype=DT, **_kwargs34)
        _bad34 = False
    except ValueError:
        _bad34 = True
    check("  missing switch, wrong-sized, or negative tail rates are refused",
          _bad34, str(_kwargs34))

# ---------------------------------------------------------------------------
# 35. ONLINE FULL-PROPOSAL VALIDATION -> DIAGONAL LATCH
# ---------------------------------------------------------------------------
print("\n[35] adaptive full proposal -> diagonal replacement")

from utils.pose_preconditioner import (
    adaptive_failure_streak,
    adaptive_loss_improved,
)

check("  a real decrease is accepted without a scene-scale threshold",
      adaptive_loss_improved(10000.0, 9999.0),
      "only float32 round-off sets the tolerance")
check("  equality, increases, and non-finite trials are rejected",
      not adaptive_loss_improved(10000.0, 10000.0)
      and not adaptive_loss_improved(10000.0, 10001.0)
      and not adaptive_loss_improved(10000.0, float("nan")),
      "a failed trial must cause the rollback")

_streak35, _switch35 = adaptive_failure_streak(0, False, 3)
_streak35, _switch35b = adaptive_failure_streak(_streak35, False, 3)
_streak35, _switch35c = adaptive_failure_streak(_streak35, False, 3)
check("  patience requires consecutive failures before switching",
      not _switch35 and not _switch35b and _switch35c and _streak35 == 3,
      "isolated full-step loss wobbles must not end acquisition")
_streak35, _switch35 = adaptive_failure_streak(_streak35, True, 3)
check("  one successful full proposal resets the failure streak",
      _streak35 == 0 and not _switch35,
      "patience counts consecutive failures, not lifetime failures")
try:
    adaptive_failure_streak(0, False, 0)
    _bad_patience35 = False
except ValueError:
    _bad_patience35 = True
check("  zero adaptive patience is refused", _bad_patience35,
      "the transition must require at least one failure")

_rates35 = torch.tensor([.003, .003, .003, .001, .001, .001], dtype=DT)
_control35 = PosePreconditioner(
    dim=6, lr=.001, target_step_norm=float(_rates35.norm()),
    refactor_every=1, max_step_mult=1.0, dtype=DT)
_adaptive35 = PosePreconditioner(
    dim=6, lr=.001, target_step_norm=float(_rates35.norm()),
    refactor_every=1, max_step_mult=1.0, adaptive_diag=True,
    diag_lr=_rates35, dtype=DT)
for _p35 in (_control35, _adaptive35):
    _p35.carry_frame(transport=None)

_g35 = torch.tensor([.8, -.4, .25, .6, -1.0, .35], dtype=DT)
_full_control35 = _control35.step(_g35)
_full_trial35 = _adaptive35.step(_g35)
_control35.refactor(); _adaptive35.refactor()
check("  adaptive acquisition is bit-identical before a rejection",
      torch.equal(_full_control35, _full_trial35),
      "the online arm cannot perturb a successful full proposal")

_replacement35 = _adaptive35.activate_adaptive_diagonal()
_want35 = -_rates35 * (_adaptive35.P_diag @ _g35)
_cap35 = float(_rates35.norm()) * _adaptive35.max_step_mult
_want35 = _want35 * min(1.0, _cap35 / max(float(_want35.norm()), 1e-30))
check("  rejection restarts m from the same gradient and applies its diagonal step",
      torch.allclose(_replacement35, _want35, atol=1e-12, rtol=1e-12)
      and torch.allclose(_adaptive35.m, (1.0 - _adaptive35.beta1) * _g35,
                         atol=0.0, rtol=0.0)
      and _adaptive35._t == 1,
      f"max|replacement-want|={float((_replacement35 - _want35).abs().max()):.2e}")
check("  the adaptive diagonal phase latches and is reported",
      float(_adaptive35._diag_w) == 1.0
      and _adaptive35.adaptive_switches == 1
      and "adaptive diagonal switches 1/1 frames" in _adaptive35.summary(),
      "requested-vs-happened must be visible in the log")

_adaptive35.carry_frame(transport=None)
check("  every new frame reacquires with the full matrix",
      float(_adaptive35._diag_w) == 0.0,
      "the previous frame's data-driven decision cannot leak")

_patient35 = PosePreconditioner(
    dim=6, lr=.001, adaptive_diag=True, adaptive_diag_patience=3, dtype=DT)
check("  adaptive patience is configured and visible in the summary",
      _patient35.adaptive_diag_patience == 3
      and "after 3 consecutive failures" in _patient35.summary(),
      "run logs must distinguish P1 from P3 transitions")

_cal35 = PosePreconditioner(
    dim=6, lr=.001, adaptive_diag=True,
    adaptive_diag_calibration_frames=4, dtype=DT)
_learned35 = [
    _cal35.finish_adaptive_calibration_frame(4, 20),
    _cal35.finish_adaptive_calibration_frame(6, 20),
    _cal35.finish_adaptive_calibration_frame(None, 10),
    _cal35.finish_adaptive_calibration_frame(8, 20),
]
check("  calibration installs the median fixed restart and diagonal boundary",
      _learned35 == [None, None, None, 7]
      and _cal35.diag_after == _cal35.restart_at == 7
      and _cal35.adaptive_diag_fixed_after == 7
      and not _cal35.adaptive_diag,
      str(_learned35))
check("  a frame that converges before switching is recorded as censored",
      _cal35._adaptive_calibration_censored == 1
      and "learned from 4 frames" in _cal35.summary()
      and "censored=1" in _cal35.summary(),
      _cal35.summary())

# A FLOOR ON THE INSTALLED TRANSITION. TUM fr1_desk calibrated to D=6 from
# p10/med/p90=3/6/8, while both the MonoGS fixed-D arm and Gaussian-SLAM's
# diag20->diag40 result argue for a LONGER coupled acquisition phase. The floor
# clamps only the installed value: the samples, the reported median and the
# censoring count must all stay exactly as they were without it, or the floor
# becomes unfalsifiable by hiding the evidence that would contradict it.
_floor35 = PosePreconditioner(
    dim=6, lr=.001, adaptive_diag=True,
    adaptive_diag_calibration_frames=4, adaptive_diag_min_iter=10, dtype=DT)
_floor_learned35 = [
    _floor35.finish_adaptive_calibration_frame(4, 20),
    _floor35.finish_adaptive_calibration_frame(6, 20),
    _floor35.finish_adaptive_calibration_frame(None, 10),
    _floor35.finish_adaptive_calibration_frame(8, 20),
]
check("  the floor raises an early median to the configured minimum",
      _floor_learned35 == [None, None, None, 10]
      and _floor35.diag_after == _floor35.restart_at == 10
      and _floor35.adaptive_diag_fixed_after == 10,
      str(_floor_learned35))
check("  the floor does not rewrite the evidence it overrode",
      _floor35.adaptive_diag_learned_median == 7
      and _floor35._adaptive_calibration_samples
      == _cal35._adaptive_calibration_samples
      and _floor35._adaptive_calibration_censored == 1
      and "FLOORED 7->10" in _floor35.summary(),
      _floor35.summary())

_nofloor35 = PosePreconditioner(
    dim=6, lr=.001, adaptive_diag=True,
    adaptive_diag_calibration_frames=2, adaptive_diag_min_iter=5, dtype=DT)
_nofloor35.finish_adaptive_calibration_frame(12, 20)
_nofloor35.finish_adaptive_calibration_frame(14, 20)
check("  a non-binding floor leaves the calibrated median alone and says so",
      _nofloor35.diag_after == 13
      and _nofloor35.adaptive_diag_learned_median == 13
      and "floor 5 not binding" in _nofloor35.summary(),
      _nofloor35.summary())

check("  the unfloored default is unchanged by the new parameter",
      _cal35.adaptive_diag_min_iter == 0 and _cal35.diag_after == 7
      and "FLOORED" not in _cal35.summary()
      and "not binding" not in _cal35.summary(),
      _cal35.summary())

for _kwargs35 in (
        {"adaptive_diag": True, "diag_after": 20},
        {"adaptive_diag": True, "ramp_from": 20, "ramp_to": 20},
        {"adaptive_diag": True, "bfgs": True},
        {"adaptive_diag_calibration_frames": -1},
        {"adaptive_diag_calibration_frames": 2},
        # A floor with no calibration to floor is a config error, not a silent
        # no-op: the user asking for it means to move a learned transition.
        {"adaptive_diag": True, "adaptive_diag_min_iter": 10},
        {"adaptive_diag": True, "adaptive_diag_calibration_frames": 4,
         "adaptive_diag_min_iter": -1}):
    try:
        PosePreconditioner(dim=6, lr=.001, dtype=DT, **_kwargs35)
        _bad35 = False
    except ValueError:
        _bad35 = True
    check("  adaptive diagonal conflicts are refused", _bad35, str(_kwargs35))

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} - " + ", ".join(FAILURES))
    sys.exit(1)

# --------------------------------------------------------------- wander ratio
# path/net separates a frame that WALKED to the answer from one that ARRIVED
# and then orbited it. Those two regimes respond to a smaller acquisition step
# in opposite directions, which is the whole reason one lr does not serve every
# scene, so the metric has to tell them apart on synthetic cases before anyone
# reads it off a run.
print("[wander] path/net separates travelling from orbiting")

_wkw = dict(dim=6, lr=1e-2, beta2=0.98, dtype=DT)

# A frame that walks: every step in the same direction. path == net exactly.
_walk = PosePreconditioner(**_wkw)
_g = torch.zeros(6, dtype=DT); _g[0] = 1.0
for _ in range(20):
    _walk.step(_g)
_walk.carry_frame()
_w = _walk.wander_spread()
assert _w is not None and _w["n"] == 1, _w
assert abs(_w["p50"] - 1.0) < 0.05, _w["p50"]
assert _w["regime"].startswith("travel-limited"), _w["regime"]
print(f"  PASS  straight-line frame reads {_w['p50']:.2f} -> {_w['regime']}")

# A frame that orbits: the gradient flips sign every iteration, so the steps
# cancel and net collapses while path keeps accumulating.
_orbit = PosePreconditioner(**_wkw)
for _i in range(20):
    _gi = torch.zeros(6, dtype=DT); _gi[0] = 1.0 if _i % 2 == 0 else -1.0
    _orbit.step(_gi)
_orbit.carry_frame()
_o = _orbit.wander_spread()
assert _o is not None and _o["p50"] > 2.0, _o["p50"]
assert _o["regime"].startswith("precision-limited"), _o["regime"]
print(f"  PASS  orbiting frame reads {_o['p50']:.2f} -> {_o['regime']}")
assert _o["p50"] > 3 * _w["p50"], (_o["p50"], _w["p50"])
print("  PASS  the two regimes are separated by more than 3x")

# Per-FRAME, and reset between frames: a walking frame after an orbiting one
# must not inherit its ratio.
_mix = PosePreconditioner(**_wkw)
for _i in range(20):
    _gi = torch.zeros(6, dtype=DT); _gi[0] = 1.0 if _i % 2 == 0 else -1.0
    _mix.step(_gi)
_mix.carry_frame()
for _ in range(20):
    _mix.step(_g)
_mix.carry_frame()
_m = _mix.wander_spread()
assert _m["n"] == 2, _m["n"]
assert min(_mix._wander_hist) < 1.5 < max(_mix._wander_hist), _mix._wander_hist
print(f"  PASS  accumulators reset per frame ({_mix._wander_hist[0]:.2f} then "
      f"{_mix._wander_hist[1]:.2f})")

# No frames observed must not raise or invent a reading.
assert PosePreconditioner(**_wkw).wander_spread() is None
print("  PASS  an unused preconditioner reports no ratio rather than a default")

# ------------------------------------------------------ online acquisition lr
# MonoGS supplies a fixed target_step_norm; SplaTAM normally supplies only lr.
# The shared online tuner must scale the active knob in both modes, always from
# the original base rather than compounding on the last scaled value.
print("[online lr] scales the active acquisition knob")
#
# The tuner INTERLEAVES the two arms on adjacent frames, so the lr in force
# changes every frame while a rung is open. The frame's iteration count is
# therefore driven by whatever the preconditioner is currently set to, exactly
# as a frontend would experience it: 30 it at the base lr, 20 at half, and a
# mildly worse 45 at a quarter (so the cascade stops at k=0.5).
def _drive(_pp, _knob, _base, _table, _frames):
    _seen = []
    for _ in range(_frames):
        _k = round(_knob(_pp) / _base, 6)
        _seen.append(_k)
        _pp._frame_iters = _table[_k]
        _pp.carry_frame()
    return _seen


_TABLE = {1.0: 30, 0.5: 20, 0.25: 45}
_olr_lr = PosePreconditioner(
    dim=6, lr=0.004, online_lr_tune=True,
    online_lr_block_frames=4, online_lr_max_halvings=4,
    online_lr_budget=200, dtype=DT)
_seen = _drive(_olr_lr, lambda p: p.lr, 0.004, _TABLE, 12)
assert _olr_lr.target_step_norm == 0.0
# block_frames=4 -> 2 pairs per look. Rung 1 (1 vs .5) decides REDUCE after 4
# frames; rung 2 (.5 vs .25) decides KEEP after 4 more; then k holds at 0.5.
assert _seen[:4] == [1.0, 0.5, 0.5, 1.0], _seen
assert _seen[4:8] == [0.5, 0.25, 0.25, 0.5], _seen
assert _seen[8:] == [0.5] * 4, _seen
assert _olr_lr.online_lr_tuner.frozen and \
    abs(_olr_lr.lr - 0.002) < 1e-15, (_olr_lr.lr, _olr_lr.online_lr_tuner.summary())
print("  PASS  SplaTAM lr alternates from the fixed base, then freezes at 0.5")

_olr_target = PosePreconditioner(
    dim=6, lr=0.001, target_step_norm=0.006,
    online_lr_tune=True, online_lr_block_frames=4,
    online_lr_max_halvings=4, online_lr_budget=100, dtype=DT)
_drive(_olr_target, lambda p: p.target_step_norm, 0.006, _TABLE, 12)
assert _olr_target.online_lr_tuner.frozen
assert abs(_olr_target.target_step_norm - 0.003) < 1e-15
assert abs(_olr_target.lr - 0.001) < 1e-15
print("  PASS  fixed-norm users still scale target_step_norm and leave lr alone")

# ------------------------------------------- per-frame convergence stand-in
# The relative gradient (|g| against the frame's own first |g|) is logged per
# frame so frames pinned at the cap, where it/frame is a constant, still carry
# a "how far did it get" reading. It is read in the existing frame-boundary
# sync, so it must survive the reset that follows it.
print("[rel-grad] per-frame convergence series")
_rg = PosePreconditioner(dim=6, lr=0.004, dtype=DT)
_g0v = torch.randn(6, dtype=DT)
for _i in range(6):                        # frame 1: gradient decays 8x
    _rg.step(_g0v * (0.5 ** (_i * 0.5)))
_rg._frame_iters = 6
_rg.carry_frame()
for _i in range(6):                        # frame 2: gradient does NOT decay
    _rg.step(_g0v * 0.9 ** 0 * (1.0 + 0.0 * _i))
_rg._frame_iters = 6
_rg.carry_frame()
_rg._frame_iters = 4                       # frame 3: ran, but never stepped
_rg.carry_frame()
_rg._frame_iters = 0                       # frame 4: did not run at all
_rg.carry_frame()
assert len(_rg._frame_rel_best) == len(_rg._frame_iter_hist) == 3, \
    (_rg._frame_rel_best, _rg._frame_iter_hist)
assert len(_rg._frame_rel_final) == 3
_b1, _b2, _b3 = _rg._frame_rel_best
assert 0.0 < _b1 < 0.5, _b1               # frame 1 got well below its first |g|
assert abs(_b2 - 1.0) < 1e-6, _b2         # frame 2 never improved on it
assert _b3 != _b3, _b3                    # no gradient -> NaN, not 0 or inf
assert _rg._frame_rel_final[0] > 0.0
print(f"  PASS  best {_b1:.3f} / {_b2:.3f} / nan aligned with it/frame; "
      f"a frame that did not run adds nothing")
_sm = PosePreconditioner(dim=6, lr=0.004, online_lr_tune=True,
                         online_lr_block_frames=4, online_lr_budget=100,
                         dtype=DT)
for _i in range(4):
    _sm.step(torch.randn(6, dtype=DT))
    _sm._frame_iters = 30
    _sm.carry_frame()
assert len(_sm.online_lr_tuner.k_log) == len(_sm._frame_rel_best) == 4
print("  PASS  the tuner's k series is aligned with the convergence series")

print("[note_frame_outcome] cap-bound fallback reaches the tuner end to end")
_cb = PosePreconditioner(
    dim=6, lr=0.004, online_lr_tune=True,
    online_lr_block_frames=12, online_lr_max_halvings=1, dtype=DT)
# GSLAM-style: fixed iteration count (100% at its own cap every frame), no
# fixed budget passed at all - only the explicit at_cap/pose_err report.
for _i in range(24):
    _cb._frame_iters = 40
    _cb.note_frame_outcome(
        at_cap=True, pose_err=(0.01 if _cb.lr == 0.004 else 0.004))
    _cb.carry_frame()
assert _cb.online_lr_tuner.frozen and abs(_cb.lr - 0.002) < 1e-15, \
    (_cb.lr, _cb.online_lr_tuner.summary())
assert _cb.online_lr_tuner.history[0][6] == "pose_err", \
    _cb.online_lr_tuner.summary()
print("  PASS  it/frame was constant throughout; pose_err alone drove REDUCE")

print("[note_frame_outcome] a no-op when no tuner is running")
_nt = PosePreconditioner(dim=6, lr=0.004, dtype=DT)
_nt.note_frame_outcome(at_cap=True, pose_err=0.5)   # must not raise
assert _nt._frame_at_cap is None and _nt._frame_pose_err is None
print("  PASS  no tuner -> note_frame_outcome does nothing")

print("[note_frame_outcome] resets between frames - no stale carryover")
_rs = PosePreconditioner(
    dim=6, lr=0.004, online_lr_tune=True,
    online_lr_block_frames=12, dtype=DT)
_rs._frame_iters = 40
_rs.note_frame_outcome(at_cap=True, pose_err=0.01)
_rs.carry_frame()
assert _rs._frame_at_cap is None and _rs._frame_pose_err is None, \
    "carry_frame() must clear both after consuming them"
_rs._frame_iters = 40                 # NEXT frame: caller forgets to report
_rs.carry_frame()
assert _rs.online_lr_tuner.k_log[-1] == _rs.online_lr_tuner.k_log[-1]  # ran
print("  PASS  a frame that skips note_frame_outcome sees None, not the "
     "previous frame's values")

# -------------------------------------- conditional anomaly metric reset
print("[anomaly carry] resets M only on an anomalous new frame")
_ar = PosePreconditioner(
    dim=2, lr=0.01, beta1=0.0, beta2=0.9,
    target_step_norm=0.2, reset_m_on_anomaly=True, dtype=DT)
_ar.carry_frame()
_ar.step(torch.tensor([1.0, 0.0], dtype=DT))
_ar.refactor(force=True)
_ar.carry_frame()
# Make the carried metric and cached inverse factor unmistakably stale.
_ar.M.copy_(torch.diag(torch.tensor([1.0, 4.0], dtype=DT)))
_ar.P.copy_(torch.diag(torch.tensor([3.0, 0.2], dtype=DT)))
_ar._have_metric = True
_ag = torch.tensor([10.0, 10.0], dtype=DT)
_ad = _ar.step(_ag)
_expected_M = torch.tensor([[100.0, 10.0], [10.0, 100.0]], dtype=DT)
_expected_d = -0.2 * _ag / _ag.norm()
assert float(_ar.anomalous()) > 0.5
assert torch.allclose(_ar.M, _expected_M, rtol=1e-12, atol=1e-12), _ar.M
assert torch.allclose(_ad, _expected_d, rtol=1e-12, atol=1e-12), _ad
assert int(_ar._anom_m_resets_dev) == 1
print("  PASS  anomalous first gradient discards carried shape, preserves "
      "trace scale, and bypasses stale P")

print("all checks passed")
