"""
pose_preconditioner.py - full-matrix adaptive preconditioning for the pose block.

THE ARGUMENT, in one paragraph.

Tracking optimises six numbers. Every SLAM system in this repo optimises them
with Adam, which keeps a DIAGONAL second moment: `v_i = E[g_i^2]`, one number
per coordinate, and never sees `g_i g_j`. That choice is not a considered one -
it is inherited from the MAP optimiser, where the parameter count makes a full
matrix impossible. Full-matrix adaptive methods (Duchi's full-matrix AdaGrad,
GGT, Shampoo) are rejected everywhere because they cost O(n^2) memory and
O(n^3) compute. At n = 6 that is a 6x6 matrix and a 6x6 eigendecomposition -
microseconds against an 11 ms render. The pose block is running an optimiser
designed under a constraint it does not have.

WHY THIS IS NOT GAUSS-NEWTON, AND WHY THAT MATTERS.

The GN line on this branch is closed (results/SECOND_ORDER_TRACKING_RECORD.md).
It died on §4.6: the renderer's analytic backward disagrees with finite
differences at cos = 0.868, FLAT across four decades of eps - they are
differentiating different functions, and no step size fixes it. That kills
anything that needs a correct J, because GN SOLVES `δ = -H^-1 g` and a wrong J
moves the FIXED POINT.

A preconditioner only reshapes: `δ = -P g` has fixed point `g = 0` for any
positive-definite P, which is exactly Adam's fixed point. A wrong P costs
convergence RATE, never the ANSWER. That is the whole reason this survives the
result that killed GN, and it is why P is estimated from the gradient stream
itself - never from a second, disagreeing path through the renderer.

`E[gg^T]` is `J^T Σ J`, the GN Hessian in expectation. So this recovers GN's
curvature information without ever forming J.

THE TRAP THAT MAKES THE OBVIOUS WIRING A NO-OP.

The tempting implementation is to transform `p.grad` in place and let the
existing Adam step run. That does NOTHING. Adam divides each coordinate by
sqrt(v_i), and v_i adapts to whatever scale the incoming gradient has, so any
DIAGONAL part of a pre-applied P is erased within a few iterations. Only the
off-diagonal mixing would survive, and the diagonal is where the measured
effect is. (It is also why the rot/trans learning-rate split works where
gradient scaling would not: an lr changes the step, and Adam's normalisation
cannot undo it.) This class therefore REPLACES the update rather than feeding
it.

MEASURED MOTIVATION, TUM fr1_desk, 250 frames, 90 tracking iterations:

    rot lr   trans lr    ATE
    0.001    0.002       3.61 cm
    0.002    0.002      17.09 cm     <- stock
    0.002    0.001     106.91 cm

dATE/dlr_rot and dATE/dlr_trans have OPPOSITE SIGNS at the stock config: a
~30x swing, and no scalar learning rate can move both blocks the way they need
to go. That is the diagonal-is-the-wrong-shape result, measured. This class is
the same argument continued - with the shape estimated from data instead of
hand-set, and the off-diagonal coupling included instead of assumed zero.

CUDA GRAPHS. `step()` is pure fixed-shape arithmetic on small tensors: no host
sync, no data-dependent branch, no allocation. It is capturable. The
eigendecomposition is NOT (cuSOLVER allocates and may sync), so it lives in
`refactor()`, which the caller invokes OUTSIDE any captured region every
`refactor_every` steps. Between refactorisations the applied preconditioner is
a constant matrix, which is exactly what a captured region needs. This is not a
compromise: the metric changes slowly, and once the cross-frame carry supplies
a good one at frame start, holding the factorisation for a stretch of
iterations costs close to nothing.

STAGE 0 vs STAGE 1. This file is written dimension-agnostic (n = dim). Stage 0
runs it at n = 7 on SplaTAM's existing (quaternion, translation) coordinates -
a feasibility probe answering "does a full matrix beat a hand-tuned diagonal at
all", for a fraction of the work. It is NOT the principled version: the
unnormalised quaternion carries a gauge direction that is an exact null
direction of the loss, and there is no adjoint in those coordinates, so the
cross-frame carry (`carry_frame`) cannot be transported and must be run with
transport=None. Stage 1 runs it at n = 6 on the SE(3) tangent, where the carry
becomes `M <- A^-T M A^-1` and the amortisation claim is available.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def adaptive_loss_improved(previous: float, current: float) -> bool:
    """Return whether a float32 loss made a numerically real decrease.

    The adaptive full-to-diagonal switch deliberately has no scene-scale
    threshold.  It only distinguishes a decrease from float32 round-off.  A
    non-finite trial is always rejected.
    """
    previous = float(previous)
    current = float(current)
    if not math.isfinite(previous) or not math.isfinite(current):
        return False
    scale = max(abs(previous), abs(current), 1.0)
    tolerance = 8.0 * np.finfo(np.float32).eps * scale
    return current < previous - tolerance


def adaptive_failure_streak(streak: int, improved: bool,
                            patience: int) -> tuple[int, bool]:
    """Advance the threshold-free full-step rejection streak."""
    patience = int(patience)
    if patience < 1:
        raise ValueError("adaptive diagonal patience must be >= 1")
    next_streak = 0 if improved else int(streak) + 1
    return next_streak, next_streak >= patience

# How far above the healthy-frame reference a frame's starting gradient
# may be and still count as normal. Above this it is excluded from the
# reference AND barred from early stopping.
#
# THE DEFAULT, AND WHY IT IS NOW A KNOB. This was a module constant, settable
# only by editing this file, and it decides which frames run the full budget -
# so it silently controls a large share of tracking cost. Reproduced on CPU in
# profiling/anom_ratchet.py: a frame that exceeds the multiple is excluded from
# the reference AND barred, and since only accepted frames update the
# reference, a SUSTAINED step change in gradient level freezes it. In that
# probe a 3x sustained shift barred 99 of the following 100 frames and the
# reference never moved again. A transient spike (5 frames) and a gradual ramp
# both recover, so the failure needs a persistent shift - which is what moving
# into a different part of a scene looks like.
_ANOM = 2.0

# Rate at which a BARRED frame is allowed to move the reference. 0.0 is the
# original behaviour: barred frames contribute nothing, which is what makes the
# freeze absorbing. A small positive value lets a persistently elevated
# gradient level be absorbed over ~1/rate frames while still lagging far behind
# a divergence spiral, whose |g_0| grows every frame rather than sitting at a
# new plateau.
_BARRED_ADMIT = 0.0

# ---------------------------------------------------------------------------
# THE STEP PROFILE (see PosePreconditioner._profile).
#
# THE BOUND THE HISTOGRAM IS READ AGAINST IS 1.0, AND THAT IS NOT A TUNED
# NUMBER. Adam is bounded per-coordinate because m and v are built from the
# SAME gradient stream: |m_i| <= sqrt(v_i), so |m_hat/sqrt(v_hat)| <= 1 and
# |d| <= lr*sqrt(n). The matrix statement is the same one - with M = E[g g^T]
# and m = E[g] the variance M - m m^T is PSD, so
#
#     |M^-1/2 m|^2 = m^T M^-1 m <= n     =>     |d| <= lr*sqrt(n) = ref
#
# so `ref` is not a scale to tune against, it is the bound the estimator
# satisfies by construction. Measured on TUM fr1 (C0, 267 Adam steps): Adam's
# max step was 0.89x ref, sitting on the bound. The metric's was 1.79x - and
# the shipped trust region sits at 10x, an order of magnitude past a bound
# that should not need enforcing.
#
# THE INTERESTING QUESTION IS THEREFORE WHY IT IS VIOLATED, and the candidates
# are all structural rather than scene-specific: m and M use different EMA
# horizons (beta1 0.9 vs beta2 0.95); m is bias-corrected and M is not; and
# under the CARRY, M describes previous frames' geometry while m describes this
# frame's trajectory, so M >= m m^T can simply fail. That is why the buckets
# are centred on 1.0 - the count above it is the violation rate.
_RAW_LO = -6                        # first bucket covers |d|/ref < 2^-5
_RAW_K = 14                         # ... last covers >= 2^7 = 128x
# Iteration bands. DIAGNOSTIC ONLY - fixed iteration counts are exactly the
# per-scene constant that must not appear in a FIX (a 40-iteration Replica
# frame and a 200-iteration TUM one do not share them). Fine for describing one
# run; the bound above carries no such constant, which is the point.
_BAND_EDGES = [5.0, 10.0, 20.0, 40.0]
_BAND_K = len(_BAND_EDGES) + 1
_BAND_NAMES = ("it0-4", "it5-9", "it10-19", "it20-39", "it40+")


# ---------------------------------------------------------------------------
# Spectrum filtering. Shared shape with the observability filter validated in
# utils/gn_tracking.py - `damp` matched lambda/(lambda + tau*lambda_max) to
# 1e-06 there, and it was the best-performing arm of that campaign. It is also
# the branch-free one, which is what keeps the applied step capturable.
# ---------------------------------------------------------------------------

def _inv_sqrt_damped(M: torch.Tensor, tau: float, eps: float,
                     shrink: float = 1.0) -> torch.Tensor:
    """((M + tau*lambda_max I)^{1/2} + eps I)^{-1}.

    THE SIGN OF THE REGULARISATION IS THE WHOLE POINT, and it is the opposite
    of the GN campaign's observability filter. That filter multiplied the STEP
    by lambda/(lambda + tau*lambda_max), suppressing motion along
    weakly-observed directions. Doing the same to M here and then INVERTING it
    would drive the smallest eigenvalues toward zero and their inverses toward
    infinity - amplifying precisely the directions that should be damped. (This
    was written that way first; the rotation-covariance and convergence tests
    caught it as a 1e20 divergence.)

    A preconditioner needs a FLOOR on the spectrum, not a shrinkage. Low
    curvature SHOULD earn a longer step - that is what preconditioning is -
    but the amount must be bounded. With lambda -> lambda + tau*lambda_max the
    ratio between the longest and shortest step is exactly

        sqrt((1 + tau) / tau)

    (about 4.6 at tau = 0.05), no matter how singular M is. Unregularised it
    would be 1/eps. That is Levenberg damping, and it degrades gracefully to
    plain gradient descent as tau grows.

    M is symmetric PSD by construction (a sum of outer products), but rank
    deficient early on - after k updates from a cold start its rank is at most
    k, so it is SINGULAR until k >= n. Inverting it directly is not optional to
    guard against; it is guaranteed to be needed. Hence the damping, which also
    supplies the regularisation the four weakly-observed DOF need (the measured
    spectrum has lambda/lambda_max = 0.010, 0.014, 0.050, 0.072, 0.742, 1.000 -
    a real 10x gap after rank 3).
    """
    # Symmetrise before eigh: M accumulates via addition of outer products and
    # stays symmetric in exact arithmetic, but float32 rounding drifts, and
    # eigh reads only one triangle - so an asymmetric drift would be silently
    # resolved differently depending on UPLO rather than raising.
    Ms = 0.5 * (M + M.transpose(-1, -2))
    Ms = Ms.double()

    # SHRINKAGE TOWARD THE DIAGONAL.  M_used = g*M + (1-g)*diag(M)
    #
    # M is 6x6 symmetric: 21 free parameters, accumulated from RANK-1 updates.
    # At a 40-iteration budget that is ~40 samples for 21 parameters - poorly
    # determined - while a diagonal needs 6. Inverting a noisy estimate of the
    # off-diagonals can cost more than the coupling they capture is worth.
    #
    # This is the standard remedy when samples are scarce relative to
    # parameters: pull toward a well-conditioned target. shrink=1 is the full
    # matrix, shrink=0 is Adam's diagonal expressed in the tangent.
    #
    # It is also the CENTRAL CLAIM made measurable. The argument for this whole
    # method is that the off-diagonal coupling matters; a sweep in `shrink`
    # tests that directly, per scene. Motivating asymmetry: on TUM at 40
    # iterations the full matrix gave 2.80-4.77 cm where Adam gave 0/5, all
    # diverged - but on Replica at 40 it LOSES to Adam. The scenes differ in
    # noise, not budget: TUM is real sensor data with ill-conditioned gradients,
    # Replica is synthetic and clean, so its diagonal may already be adequate
    # and the extra 15 parameters mostly estimation noise.
    if shrink < 1.0:
        Ms = shrink * Ms + (1.0 - shrink) * torch.diag(torch.diagonal(Ms))
    n = Ms.shape[-1]
    I = torch.eye(n, dtype=Ms.dtype, device=Ms.device)
    # Checked BEFORE the ridge is added, or the ridge itself becomes the whole
    # spectrum and the cold start returns 1/sqrt(ridge) instead of identity.
    if float(Ms.abs().max()) <= 0.0:
        return torch.eye(n, dtype=M.dtype, device=M.device)
    # M is rank-1 after the first step and rank <= k after k, so it arrives
    # here with (n - k) EXACTLY repeated zero eigenvalues. LAPACK's
    # divide-and-conquer syevd fails outright on that (error code 5) rather
    # than returning the degenerate basis - it is not an edge case here, it is
    # every cold start. A ridge scaled to the trace breaks the degeneracy
    # without moving any eigenvalue that carries information; the damping
    # below would suppress a perturbation this size regardless.
    scale = float(torch.diagonal(Ms, dim1=-2, dim2=-1).sum()) / n
    ridge = max(scale, 1.0) * 1e-14
    for _ in range(4):
        try:
            evals, evecs = torch.linalg.eigh(Ms + ridge * I)
            break
        except Exception:
            ridge *= 1e3
    else:
        # Four escalations failed: fall back to no preconditioning rather than
        # propagating a bad metric into the step. Momentum SGD is a defensible
        # thing to do when the metric cannot be trusted.
        return torch.eye(n, dtype=M.dtype, device=M.device)
    # Numerical negatives on a PSD matrix are rounding, not information.
    evals = evals.clamp_min(0.0)
    lmax = evals.max()
    if float(lmax) <= 0.0:
        return torch.eye(n, dtype=M.dtype, device=M.device)
    floored = evals + tau * lmax
    inv = 1.0 / (floored.sqrt() + eps)
    P = (evecs * inv.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return P.to(M.dtype)


def _inv_sqrt_damped_diag(M: torch.Tensor, tau: float,
                          eps: float) -> torch.Tensor:
    """Exact ``shrink=0`` factor without an unnecessary eigendecomposition."""
    diagonal = torch.diagonal(0.5 * (M + M.transpose(-1, -2))).double()
    n = diagonal.numel()
    if float(diagonal.abs().max()) <= 0.0:
        return torch.eye(n, dtype=M.dtype, device=M.device)
    # Match _inv_sqrt_damped's degeneracy-breaking ridge exactly. For a
    # diagonal matrix its eigenvalues are the diagonal entries themselves, so
    # the remaining inverse square root is elementwise.
    scale = float(diagonal.sum()) / n
    ridge = max(scale, 1.0) * 1e-14
    evals = (diagonal + ridge).clamp_min(0.0)
    lmax = evals.max()
    if float(lmax) <= 0.0:
        return torch.eye(n, dtype=M.dtype, device=M.device)
    inv = 1.0 / ((evals + tau * lmax).sqrt() + eps)
    return torch.diag(inv).to(M.dtype)


# ---------------------------------------------------------------------------
# SE(3) adjoint, for the Stage 1 cross-frame carry.
# ---------------------------------------------------------------------------

def se3_adjoint(T: torch.Tensor) -> torch.Tensor:
    """Ad_T for twists ordered [rho(3) translation ; theta(3) rotation].

        Ad = [[R, [t]x R],
              [0, R    ]]

    NOT orthogonal - the [t]x R block is what makes transport a real correction
    rather than a relabelling. Under pure rotation it degenerates to a
    block-diagonal rotation of the metric; under translation it genuinely
    reshapes it, which is the case that matters between consecutive frames.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    tx = torch.zeros((3, 3), dtype=T.dtype, device=T.device)
    tx[0, 1], tx[0, 2] = -t[2], t[1]
    tx[1, 0], tx[1, 2] = t[2], -t[0]
    tx[2, 0], tx[2, 1] = -t[1], t[0]
    Ad = torch.zeros((6, 6), dtype=T.dtype, device=T.device)
    Ad[:3, :3] = R
    Ad[:3, 3:] = tx @ R
    Ad[3:, 3:] = R
    return Ad


def transport_metric(M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Carry a second-moment matrix across a change of tangent frame.

    With left perturbation `T <- exp(dxi) T`, dxi is expressed in CAMERA-frame
    coordinates, so M from the previous frame is stated in a different
    coordinate system than the current one. Copying it mixes the two.

    Twists are contravariant, `xi_t = A xi_{t-1}`; gradients are covectors, so
    `g_t = A^-T g_{t-1}`; and therefore

        M_t = E[g_t g_t^T] = A^-T M_{t-1} A^-1.

    Solved rather than inverted: A is well conditioned for a rigid transform,
    but forming A^-1 explicitly to multiply twice is both slower and less
    accurate than two triangular solves.
    """
    # A^-T M A^-1  ==  (A^-T) M (A^-T)^T, and X = A^-T M is solve(A^T, M).
    X = torch.linalg.solve(A.transpose(-1, -2), M)
    return torch.linalg.solve(A.transpose(-1, -2), X.transpose(-1, -2)).transpose(-1, -2)


# ---------------------------------------------------------------------------
# DAMPED BFGS ON THE SE(3) TANGENT.
#
# WHY, after the sample metric closed. E[gg^T] answers "how are the gradients I
# have seen distributed?" - a gradient COVARIANCE, not curvature. The Level 2
# result says that estimating that object better buys nothing, because an EMA
# over ~20 iterations already estimates it well. So the remaining move is to use
# a DIFFERENT KIND OF INFORMATION, not a better estimate of the same one.
#
# The secant pair is the natural candidate and it is free:
#
#     s_k = xi_{k+1} - xi_k        y_k = g_{k+1} - g_k        y_k = H s_k
#
# for a locally quadratic objective. The step is already known and the next
# backward is already paid for - no renderer J, no finite differences, no second
# render. At n = 6 the full inverse-Hessian approximation is 36 floats.
#
# AND IT SURVIVES SS4.6 FOR THE REASON THE PRECONDITIONER DID. delta = -eta B g
# has fixed point g = 0 for any positive-definite B, so a wrong B costs
# convergence RATE, never the ANSWER. This is NOT Gauss-Newton: nothing is
# solved, only reshaped.
#
# WHAT MAKES IT SAFE HERE. Vanilla BFGS assumes y^T s > 0, which a noisy
# non-convex SLAM loss does not guarantee. Every pair is screened, and the
# spectrum of B is clamped so no single bad pair can produce an enormous step.
# ---------------------------------------------------------------------------

def bfgs_update(B: torch.Tensor, s: torch.Tensor, y: torch.Tensor,
                curv_eps: float = 1e-8) -> torch.Tensor:
    """One damped BFGS inverse update, returned as a new matrix.

        B <- (I - rho s y^T) B (I - rho y s^T) + rho s s^T,   rho = 1/(y^T s)

    THE PAIR IS SCREENED, NOT TRUSTED. The update is applied only when

        y^T s > curv_eps * |y| * |s|

    i.e. the observed gradient change has a positive component along the step
    that is not merely numerical. A pair failing this describes NEGATIVE
    curvature along s, where the secant equation carries no usable information
    and applying it would destroy the positive-definiteness the fixed-point
    argument depends on. Rejected pairs leave B untouched, which is the
    standard skip rule and the conservative choice: a stale B is a worse
    preconditioner, a non-PD B is a wrong optimiser.

    Branch-free via torch.where so the body stays CUDA-graph capturable.
    """
    n = B.shape[0]
    I = torch.eye(n, dtype=B.dtype, device=B.device)
    sy = y.dot(s)
    ok = sy > curv_eps * y.norm() * s.norm()
    # Guard the reciprocal itself, not just the result: sy can be 0 or negative
    # on a rejected pair, and inf/nan would propagate through the where.
    rho = 1.0 / torch.where(ok, sy, torch.ones_like(sy))
    V = I - rho * torch.outer(s, y)
    B_new = V @ B @ V.transpose(0, 1) + rho * torch.outer(s, s)
    return torch.where(ok, B_new, B)


def clamp_spectrum(B: torch.Tensor, max_cond: float,
                   eps: float = 1e-30) -> torch.Tensor:
    """Symmetrise and clamp eigenvalues to [lambda_max / max_cond, lambda_max].

    THE SAME JOB tau DOES FOR M, done where BFGS needs it. B approximates
    H^-1 DIRECTLY - it is the preconditioner, not something to be inverted - so
    a near-zero eigenvalue is harmless but a runaway one is a runaway step, and
    accumulated float error can drift B out of symmetry over hundreds of
    updates. Clamping bounds the condition number and restores symmetry in the
    same 6x6 eigendecomposition.

    NOT capturable (eigh allocates and may sync) - call it from refactor().
    """
    B = 0.5 * (B + B.transpose(0, 1))
    # A NON-FINITE B MUST NOT REACH eigh. cuSOLVER raises on inf/nan rather
    # than returning them, which turns a recoverable numerical event into a
    # dead run - it killed one at frame 225 of 250. Falling back to the
    # identity loses the accumulated curvature for that frame, which is the
    # correct trade: the alternative is no run at all.
    if not bool(torch.isfinite(B).all()):
        return torch.eye(B.shape[0], dtype=B.dtype, device=B.device)
    ev, Q = torch.linalg.eigh(B)
    hi = ev.max().clamp_min(eps)
    ev = ev.clamp(min=hi / max_cond, max=hi)
    return (Q * ev) @ Q.transpose(0, 1)


# ---------------------------------------------------------------------------
# STAGE 1: preconditioning on the SE(3) tangent, without re-parameterising the
# tracker.
#
# Stage 0 preconditioned SplaTAM's 7 raw numbers (unnormalised quaternion +
# translation) and produced a new pathology at every fix: unbounded opening
# steps, attraction to the gauge direction, and the gauge crowding the trust
# region. Three of those four are consequences of preconditioning a
# 7-parameter representation of a 6-dimensional object. The tangent has no
# gauge freedom, so none of them exist here.
#
# THE CHEAP WAY IN. Re-parameterising the tracker means threading a delta-xi
# through transform_to_frame, the candidate save/restore, the early-stop pose
# deltas and the graph capture. None of that is necessary: the render path can
# stay exactly as it is, and only the STEP moves to the tangent.
#
#   1. autograd fills grads on (q, t) as it already does
#   2. map that 7-vector to the 6-D tangent with the exact analytic Jacobian
#      of the parameterisation map (below)
#   3. precondition in 6-D - full-rank M, no null direction
#   4. apply the step as T <- exp(dxi) T and write (q, t) back
#
# AND THIS JACOBIAN IS NOT THE ONE THAT KILLED GAUSS-NEWTON. §4.6 of the record
# is about the RENDERER's backward disagreeing with finite differences of the
# rendered loss. This is the Jacobian of a closed-form algebraic map between
# two parameterisations of the same pose. It is exact, it is checkable against
# finite differences to machine precision, and the test does exactly that.
#
# Convention: left perturbation T' = exp(dxi) T on a world-to-camera pose,
# dxi = [rho(3) translation ; theta(3) rotation], matching se3_exp above and
# MonoGS's update_pose - so the metric M lives in the same tangent the adjoint
# transport is written for.
# ---------------------------------------------------------------------------

def quat_trans_grad_to_tangent(q, t, g_q, g_t):
    """Map dL/d(q, t) to dL/d(dxi) at dxi = 0.

    First order in the perturbation, with T' = exp(dxi) T:

        R' = (I + [theta]x) R          t' = t + [theta]x t + rho

    so dt/drho = I and dt/dtheta = -[t]x (because [theta]x t = -[t]x theta).

    For the rotation, R' = R_delta R is a LEFT multiply, so in quaternions
    q' = q_delta (x) q with q_delta ~ [1, theta/2]. Expanding the Hamilton
    product and keeping first order, with q = (w, v):

        dq_w/dtheta = -(1/2) v^T
        dq_v/dtheta =  (1/2) (w I - [v]x)

    dq/drho is zero - translating the camera does not rotate it.

    NOTE ON THE GAUGE. The loss sees normalize(q), so dL/dq_unnormalised
    already carries no useful component along q itself. dq/dtheta above is
    orthogonal to q by construction (a rotation cannot change the norm), so
    the contraction discards the gauge component automatically. That is the
    whole Stage 0 problem, gone by construction rather than by projection.
    """
    w = q[0]
    v = q[1:]

    dqw_dtheta = -0.5 * v                       # (3,)
    # (1/2)(w I - [v]x)
    dqv_dtheta = 0.5 * (w * torch.eye(3, dtype=q.dtype, device=q.device)
                        - _skew3(v))            # (3,3), row i = d q_v[i] / d theta

    g_theta = dqw_dtheta * g_q[0] + dqv_dtheta.transpose(0, 1) @ g_q[1:]
    # (dt/dtheta)^T g_t = (-[t]x)^T g_t = [t]x g_t
    g_theta = g_theta + _skew3(t) @ g_t
    g_rho = g_t
    return torch.cat([g_rho, g_theta])


# ---------------------------------------------------------------------------
# THE SAMPLE METRIC. One backward, many samples.
#
# THE PROBLEM IT ATTACKS. M is 6x6 symmetric - 21 free parameters - and the
# EMA above feeds it ONE rank-1 update per iteration. A 40-iteration frame
# therefore supplies ~40 samples for 21 parameters, and beta2=0.95 discards
# most of them again. That is the contradiction at the centre of this method:
# the estimator needs tracking iterations to learn a metric whose whole purpose
# is to remove tracking iterations. It also means M reaches full rank only
# after >= 6 iterations, and until then the step is being taken through a
# rank-deficient metric propped up by the Levenberg floor.
#
# THE OBSERVATION. `g` is not a measurement. It is a SUM of N measurements that
# the backward already computed and then destroyed:
#
#     g = sum_i g_i,   g_i = the pose gradient contributed by Gaussian i
#
# SplaTAM is ISOTROPIC on TUM (log_scales.shape[1] == 1, so transform_to_frame
# leaves unnorm_rotations alone), which means the camera pose reaches the loss
# through EXACTLY ONE tensor: transformed_pts, the [N,3] Gaussian centres in
# camera frame. Retaining its grad costs nothing and hands back all N
# per-Gaussian contributions. Then
#
#     M_inst = G^T G,   G = [N,6]
#
# is a single 6xNx6 gemm - microseconds against an 11 ms render - and it is
# full rank from the FIRST backward of the FIRST frame.
#
# WHAT THIS IS AND IS NOT. sum_p g_p g_p^T over RESIDUALS (pixels) is the
# empirical Fisher, which is J^T Sigma J in expectation - the object the module
# docstring above appeals to. Grouping by GAUSSIAN is not that: each g_i has
# already summed the pixels that Gaussian touched, so this is a sum of squares
# of partial sums under a grouping the renderer chose. It sits strictly between
# one-sample-per-iteration and the per-pixel Fisher, and it is the only one of
# the three that costs no CUDA work. Do not describe it as the Fisher.
# ---------------------------------------------------------------------------

def per_gaussian_tangent_grads(pts_cam: torch.Tensor,
                               grad_pts_cam: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian tangent gradients [N,6] from dL/d(camera-frame centres).

    Same convention as `quat_trans_grad_to_tangent`: left perturbation
    T' = exp(dxi) T, dxi = [rho ; theta].

    For a centre p_c = R p_w + t, that perturbation gives

        p_c' = p_c + [theta]x p_c + rho

    so dp_c/drho = I and dp_c/dtheta = -[p_c]x. Contracting with u_i = dL/dp_ci:

        g_rho,i   = u_i
        g_theta,i = (-[p_ci]x)^T u_i = [p_ci]x u_i = p_ci x u_i

    THE INVARIANT THAT MAKES THIS CHECKABLE WITHOUT A GPU:

        per_gaussian_tangent_grads(...).sum(0) == quat_trans_grad_to_tangent(...)

    to float precision, whenever |q| = 1 and the Gaussians are isotropic. Both
    sides are the same total gradient; only the grouping differs. Test 14 in
    utils/test_pose_preconditioner.py asserts it, which is why no GPU minute
    needs to be spent finding out whether the decomposition is right.
    (The identity holds because sum_i [R p_wi + t]x u_i splits into the
    rotation term, which the quaternion path carries, and [t]x sum_i u_i,
    which is the `_skew3(t) @ g_t` term there.)
    """
    return torch.cat([grad_pts_cam,
                      torch.cross(pts_cam, grad_pts_cam, dim=1)], dim=1)


def sample_metric(G: torch.Tensor, trace_match: bool = True) -> torch.Tensor:
    """G^T G from a [N, dim] sample matrix, optionally trace-matched to g g^T.

    TRACE MATCHING IS NOT COSMETIC - it is what makes this a SINGLE-VARIABLE
    change. trace(g g^T) = |g|^2, while trace(G^T G) = sum_i |g_i|^2. The step
    is -lr * M^{-1/2} m, so an unmatched swap rescales every step and PRE_LR,
    tau and max_step_mult all silently mean something else - the arm would
    measure a learning-rate change wearing a metric's clothes.

    THE RATIO IS NOT ~N. It is

        n_eff = sum_i |g_i|^2 / |sum_i g_i|^2

    and it measures CANCELLATION, not sample count:

        mutually incoherent (random directions)       n_eff ~ 1
        all g_i identical - one gradient in disguise  n_eff ~ 1/N
        heavy cancellation, near-stationary           n_eff up to N

    THIS WAS DOCUMENTED BACKWARDS FOR ONE COMMIT and a go/no-go rule was built
    on it ("n_eff near 1 means the decomposition carries nothing"). The exact
    opposite is true: near 1 is the INCOHERENT case, which is the regime where
    G^T G has six comparable eigenvalues. The synthetic check that measured
    1.64 on random data was that correction arriving early and being read as a
    curiosity instead. TUM fr1 measures 0.3 - coherence 1/0.3 = 3.3 against a
    maximum of N ~ 1e5, i.e. overwhelmingly incoherent with a small systematic
    component.

    n_eff IS THEREFORE NOT THE GO/NO-GO DIAGNOSTIC. It says the rescaling is
    real and must be divided out; it says nothing about whether the metric's
    SHAPE differs from a rank-1 one. `rank1_alignment` answers that, and it is
    the same quantity test group 14 already asserts.

    Rescaling to trace(M_inst) = |g|^2 leaves the SHAPE of the metric - its
    eigenvectors and its relative eigenvalues - as the only thing that differs
    from the rank-1 path. That shape is the variable under test.
    """
    M = G.transpose(0, 1) @ G
    if trace_match:
        g = G.sum(0)
        M = M * (g.dot(g) / torch.diagonal(M).sum().clamp_min(1e-30))
    return M


def effective_samples(G: torch.Tensor) -> torch.Tensor:
    """sum_i |g_i|^2 / |sum_i g_i|^2 - how many INDEPENDENT samples a backward
    is worth.

    Reads ~1 when the per-Gaussian gradients are mutually INCOHERENT and ~1/N
    when they are all the same vector. It is the factor sample_metric() divides
    out, and worth watching for that reason - but it is NOT the go/no-go number
    for this arm, whatever an earlier version of this docstring claimed. See
    rank1_alignment.

    Kept as a tensor - reading it is a host sync, so callers must sample it
    occasionally rather than every iteration.
    """
    g = G.sum(0)
    return G.pow(2).sum() / g.dot(g).clamp_min(1e-30)


def rank1_alignment(G: torch.Tensor) -> torch.Tensor:
    """cos(G^T G, g g^T) in the Frobenius inner product, g = G.sum(0).

    THE ACTUAL GO/NO-GO NUMBER FOR THIS ARM, and the one n_eff was mistakenly
    asked to be. It answers the only question that matters: does the
    per-Gaussian decomposition produce a metric whose SHAPE differs from the
    rank-1 one the shipped path already builds?

        ~1.0   G^T G is a rescaled g g^T. The decomposition carries no
               direction the sum did not, the arms cannot separate, and no
               amount of tuning will change that. STOP.
        < 1    the sample metric spans directions g g^T cannot reach - which is
               the entire premise, since g g^T is rank 1 and G^T G is rank 6.

    Scale-free by construction, so trace matching does not affect it. Test
    group 14 asserts exactly this quantity on synthetic data, where it is 0.52.

    Kept as a tensor - reading it is a host sync, so sample it occasionally
    rather than every iteration.
    """
    M = G.transpose(0, 1) @ G
    g = G.sum(0)
    R = torch.outer(g, g)
    return (M * R).sum() / (M.norm() * R.norm()).clamp_min(1e-30)


def apply_tangent_step(q, t, dxi, mat_to_quat):
    """T <- exp(dxi) T, returned as (q, t). q comes back unit-norm.

    `mat_to_quat` is injected rather than imported so this file stays free of
    a dependency on the tracker's conventions - the caller passes whichever
    rotation-to-quaternion routine matches its own build_rotation.
    """
    # Imported, not reimplemented: gn_tracking's se3_exp is validated to a
    # 2e-16 round trip from 1e-9 through pi, with the series branch widened to
    # 1e-2 because the closed forms cancel catastrophically below that - and
    # accepted steps here measure ~1e-4, squarely in the band where a naive
    # implementation is wrong. Lazy so this module has no import-order
    # dependency on the tracker's package layout.
    T = torch.eye(4, dtype=q.dtype, device=q.device)
    T[:3, :3] = _quat_to_mat(q)
    T[:3, 3] = t
    T_new = se3_exp_capturable(dxi.to(q.dtype)) @ T
    q_new = mat_to_quat(T_new[:3, :3])
    return q_new / q_new.norm().clamp_min(1e-20), T_new[:3, 3]


def se3_exp_capturable(xi):
    """Branch-free se3_exp. Same map, no host sync, no Python branch.

    gn_tracking's se3_exp does `ang = float(theta.norm())` and branches on it to
    pick the Taylor series near zero. That is a host readback AND a
    data-dependent branch, so a CUDA graph capture freezes whichever branch it
    happened to record - the tracker would then use the wrong series for every
    replay. This computes both and selects with torch.where.

    The unselected branch can be NaN at ang -> 0 (0/0 in the closed forms);
    torch.where discards it elementwise, and the denominators are clamped so
    it cannot poison the selected side. Safe here because this runs under
    no_grad - torch.where WOULD propagate NaN through a backward pass.
    """
    rho, theta = xi[:3], xi[3:]
    a = theta.norm()
    a2 = a * a
    a_s = a.clamp_min(1e-12)
    s, c = torch.sin(a), torch.cos(a)
    small = a < 1e-2                      # matches gn_tracking's widened band

    A = torch.where(small, 1.0 - a2 / 6.0, s / a_s)
    B = torch.where(small, 0.5 - a2 / 24.0, (1.0 - c) / (a_s * a_s))
    C = torch.where(small, 1.0 / 6.0 - a2 / 120.0, (a - s) / (a_s * a_s * a_s))

    W = _skew3(theta)
    W2 = W @ W
    I = torch.eye(3, dtype=xi.dtype, device=xi.device)
    R = I + A * W + B * W2
    V = I + B * W + C * W2

    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log_capturable(T):
    """Branch-free SE(3) log. The inverse of se3_exp_capturable.

    Same structure and the same reason as the exp: both series are computed and
    selected with torch.where, so a capture cannot freeze the wrong branch and
    nothing is read to the host.

    WHY THIS EXISTS (C0). step() records the tangent step it REQUESTS, in
    _step_sum. During a handoff, Adam moves the pose and step() never runs, so
    the Adam phase's step size has never been measured at all - and that is
    exactly the quantity that decides whether the handoff supplies DIRECTION or
    DISTANCE. Differencing the pose across an Adam step and taking this log
    puts that motion in the SAME units as _step_sum. A 7-parameter (q, t) delta
    norm is a different quantity and is not comparable to a 6-vector tangent
    norm, so the obvious cheap version of this measurement would have answered
    a different question.

    Unlike the exp, every denominator here is clamped rather than left to be
    discarded by torch.where, so the unselected branch is finite instead of
    NaN. Same result, one less thing to reason about.

    SINGULAR AT a = pi, where sin a -> 0 with the rotation still well defined.
    Out of range by construction: this only ever sees the relative motion of
    one tracking iteration, which is ~1e-4 rad. The round-trip test pins the
    band actually used and does not claim the rest.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    # Clamped because a rotation matrix that has drifted by 1e-7 - which
    # composing two normalised quaternions easily produces - pushes the
    # argument outside [-1, 1] and NaNs the entire measurement.
    cos_a = ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0) * 0.5
    a = torch.acos(cos_a.clamp(-1.0, 1.0))
    a2 = a * a
    s = torch.sin(a)
    small = a < 1e-2                      # matches se3_exp_capturable's band

    # theta = k * vee(R - R^T),  k = a / (2 sin a),  series 1/2 + a^2/12 + ...
    # The a^4 terms in both series are carried because without them the
    # round-trip error at the branch boundary is 2.5e-10 - fine for the ~1e-4
    # steps this actually sees, but a discontinuity at a = 1e-2 that someone
    # would eventually have to rule out. With them it is ~1e-14 everywhere.
    k = torch.where(small, 0.5 + a2 / 12.0 + 7.0 * a2 * a2 / 720.0,
                    0.5 * a / s.clamp_min(1e-12))
    theta = k * torch.stack([R[2, 1] - R[1, 2],
                             R[0, 2] - R[2, 0],
                             R[1, 0] - R[0, 1]])

    # rho = V^-1 p,  V^-1 = I - W/2 + c2 W^2
    #   c2 = 1/a^2 - (1 + cos a) / (2 a sin a),  series 1/12 + a^2/720 + ...
    c2 = torch.where(small,
                     1.0 / 12.0 + a2 / 720.0 + a2 * a2 / 30240.0,
                     1.0 / a2.clamp_min(1e-24)
                     - (1.0 + cos_a) / (2.0 * a.clamp_min(1e-12)
                                        * s.clamp_min(1e-12)))
    W = _skew3(theta)
    I = torch.eye(3, dtype=T.dtype, device=T.device)
    Vinv = I - 0.5 * W + c2 * (W @ W)
    return torch.cat([Vinv @ p, theta])


def tangent_of_pose_delta(q0, t0, q1, t1):
    """The dxi with exp(dxi) T0 == T1, for poses given as (quaternion, t).

    LEFT multiplication, matching apply_tangent_step's `T <- exp(dxi) T`, so
    what this returns is directly comparable to what step() returns.

    BOTH QUATERNIONS ARE NORMALISED FIRST, and that is not tidiness.
    cam_unnorm_rots is UNNORMALISED and Adam moves its norm freely; the norm is
    an exact gauge direction the loss cannot see (see note_gauge). Measuring it
    would report gauge drift as tracking motion, and since the gauge drift is
    precisely what the preconditioner path renormalises away, the comparison
    would be biased in favour of the phase being measured.

    COMPUTED IN float64 WHATEVER COMES IN, AND THAT IS LOAD-BEARING, NOT
    HYGIENE. This differences two absolute poses to recover a relative motion
    of ~1e-4, which is catastrophic cancellation twice over: dR = R1 R0^T is a
    near-identity built from products of O(1) numbers, and t1 - dR t0 subtracts
    two O(1) translations. In float32 (eps 1.2e-7) the surviving signal is
    about three digits at |dxi| = 1e-3 and NONE at 1e-5.

        MEASURED, worst relative error over 400 random poses:
            |dxi|     float32     float64
            1e-5      3.9e+00     1.8e-08
            1e-4      9.8e-01     7.9e-10
            1e-3      1.2e-01     5.6e-11

    The pose params are float32, and 1e-4 is exactly the band tracking steps
    live in - so the obvious version of this function returns ~100% error on
    every number it was written to produce, while looking entirely reasonable
    in a log. Seven scalars promoted to double costs nothing.
    """
    _dt = torch.float64
    q0, t0 = q0.to(_dt), t0.to(_dt)
    q1, t1 = q1.to(_dt), t1.to(_dt)
    R0 = _quat_to_mat(q0)
    R1 = _quat_to_mat(q1)
    dR = R1 @ R0.transpose(0, 1)
    T = torch.eye(4, dtype=_dt, device=q0.device)
    T[:3, :3] = dR
    T[:3, 3] = t1 - dR @ t0
    return se3_log_capturable(T)


def _quat_to_mat_batch(q):
    """(..., 4) (w, x, y, z) -> (..., 3, 3). Same formula as _quat_to_mat."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    w, x, y, z = q.unbind(-1)
    r0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                      2 * (x * z + w * y)], dim=-1)
    r1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                      2 * (y * z - w * x)], dim=-1)
    r2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                      1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def _se3_log_batch(R, p):
    """(..., 3, 3), (..., 3) -> (..., 6). se3_log_capturable over a batch.

    Identical series, clamps and branch band; only the indexing is batched.
    """
    cos_a = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    a = torch.acos(cos_a.clamp(-1.0, 1.0))
    a2 = a * a
    s = torch.sin(a)
    small = a < 1e-2
    k = torch.where(small, 0.5 + a2 / 12.0 + 7.0 * a2 * a2 / 720.0,
                    0.5 * a / s.clamp_min(1e-12))
    theta = k[..., None] * torch.stack([R[..., 2, 1] - R[..., 1, 2],
                                        R[..., 0, 2] - R[..., 2, 0],
                                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    c2 = torch.where(small,
                     1.0 / 12.0 + a2 / 720.0 + a2 * a2 / 30240.0,
                     1.0 / a2.clamp_min(1e-24)
                     - (1.0 + cos_a) / (2.0 * a.clamp_min(1e-12)
                                        * s.clamp_min(1e-12)))
    tx, ty, tz = theta.unbind(-1)
    o = torch.zeros_like(tx)
    W = torch.stack([torch.stack([o, -tz, ty], dim=-1),
                     torch.stack([tz, o, -tx], dim=-1),
                     torch.stack([-ty, tx, o], dim=-1)], dim=-2)
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    Vinv = I - 0.5 * W + c2[..., None, None] * (W @ W)
    return torch.cat([(Vinv @ p[..., None])[..., 0], theta], dim=-1)


def tangent_of_pose_delta_batch(q0, t0, q1, t1):
    """tangent_of_pose_delta over a batch: (N, 4), (N, 3) x 2 -> (N, 6) float64.

    WHY IT EXISTS. The stopping bookkeeping called the single-row version once
    per drained row, inside a Python loop, on seven-element CPU tensors. Each
    call dispatches a few dozen tiny float64 ops; timed on CPU that is ~1.3 ms
    per row, i.e. ~1.3 ms added to EVERY tracking iteration while the GPU sits
    idle waiting for the host. On GSLAM that is roughly a quarter of the
    per-iteration cost, and it is a large part of why stopping cut 31% of
    iterations but only ~5% of tracking time on fr1_desk.

    Same float64 promotion, same formulas, same branch band as the single-row
    function - which stays as it is, since it is also used where batching does
    not apply. utils/test_tangent_batch.py pins the two against each other.
    """
    _dt = torch.float64
    q0, t0 = q0.to(_dt), t0.to(_dt)
    q1, t1 = q1.to(_dt), t1.to(_dt)
    R0 = _quat_to_mat_batch(q0)
    R1 = _quat_to_mat_batch(q1)
    dR = R1 @ R0.transpose(-1, -2)
    p = t1 - (dR @ t0[..., None])[..., 0]
    return _se3_log_batch(dR, p)


def mat_to_quat_capturable(R):
    """Branch-free rotation -> (w, x, y, z).

    gn_tracking's mat_to_quat uses Shepperd's method, which picks a branch by
    comparing trace terms - four data-dependent branches and a host sync each.
    This computes all four magnitudes from the diagonal and recovers the signs
    from the off-diagonals, which needs no branching at all.

    Less accurate than Shepperd near 180 degrees, where the near-zero component
    is recovered from a difference of similar numbers. Irrelevant here: this
    only ever sees exp(dxi) applied to a tracking increment, where the rotation
    is a fraction of a degree and w is essentially 1. The round-trip test pins
    it against the Shepperd implementation over the range actually used.
    """
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    w = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)) * 0.5
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 0.5
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) * 0.5
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) * 0.5

    # Signs from the off-diagonals. torch.sign returns 0 at exactly 0, which
    # would silently zero a component, so select +/-1 explicitly.
    def _sgn(d):
        return torch.where(d >= 0, torch.ones_like(d), -torch.ones_like(d))

    x = x * _sgn(R[2, 1] - R[1, 2])
    y = y * _sgn(R[0, 2] - R[2, 0])
    z = z * _sgn(R[1, 0] - R[0, 1])
    q = torch.stack([w, x, y, z])
    return q / q.norm().clamp_min(1e-20)


def _skew3(v):
    o = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([o, -v[2], v[1]]),
        torch.stack([v[2], o, -v[0]]),
        torch.stack([-v[1], v[0], o]),
    ])


def _quat_to_mat(q):
    """(w, x, y, z) -> 3x3, matching SplaTAM's build_rotation."""
    q = q / q.norm().clamp_min(1e-20)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


# ---------------------------------------------------------------------------

class PosePreconditioner:
    """Full-matrix adaptive preconditioner over a flat n-vector of pose params.

    Usage per tracking iteration:

        pre.step(grad_flat)          # capturable: fixed shapes, no sync
        pre.refactor()               # OUTSIDE any captured region

    and per frame:

        pre.carry_frame(transport=A) # or transport=None to reset
    """

    def __init__(self, dim: int, lr: float,
                 beta1: float = 0.9,
                 # 0.95, NOT Adam's 0.999. The default horizon is ~1000 steps,
                 # which inside a 40-90 iteration frame never adapts at all -
                 # M would stay frozen at whatever the carry supplied, and a
                 # move into new geometry would leave a stale metric for many
                 # frames. The carry supplies the INITIALISATION; beta2
                 # supplies the within-frame refinement, and that wants a
                 # horizon of tens of iterations, not thousands.
                 beta2: float = 0.95,
                 tau: float = 0.05,     # damping, from the GN campaign's best arm
                 # 1.0 = full matrix, 0.0 = diagonal (Adam in the tangent).
                 # See _inv_sqrt_damped: 21 parameters from rank-1 updates is
                 # a poor estimate at short budgets, and this is also the
                 # sweep that measures whether the coupling earns its keep.
                 shrink: float = 1.0,
                 # Estimate M from the PER-GAUSSIAN gradient decomposition the
                 # backward already computes, instead of one rank-1 per
                 # iteration. This object only records the flag and consumes
                 # the reduced matrix through step(M_inst=...) - the caller
                 # owns the decomposition, because only the caller knows how
                 # its renderer routes the pose. See per_gaussian_tangent_grads.
                 sample_metric: bool = False,
                 stop_improve: float = 0.01,
                 stop_anchor: str = "min",
                 # ISOTROPIC PRIOR AT FRAME START, instead of M = 0.
                 #
                 #     M_0 = (|g_0|^2 / n) I     so trace(M_0) = |g_0|^2
                 #
                 # THE CLEAN NO-CARRY EXPERIMENT. Starting from zero makes the
                 # two arms differ in ways that have nothing to do with the
                 # estimator: the rank-1 arm is SINGULAR for its first n steps
                 # and leans on the Levenberg floor, while the sample arm is
                 # full rank from its first backward. So a carry-off comparison
                 # measures rank, not shape - which is the thing under test.
                 #
                 # Seeding both arms with the same full-rank, correctly-scaled
                 # isotropic prior removes that: identical initial scale (the
                 # trace matches the trace-matching convention exactly),
                 # identical rank, no temporal carry in either. What remains
                 # different is SHAPE alone.
                 #
                 # It also removes the need for the cold-EMA bias correction -
                 # M starts at the right magnitude, so 1/(1-beta2^t) would
                 # over-inflate it. _M_cold is cleared when this is on.
                 m0_iso: bool = False,
                 # SECANT MODE. Replace M = E[gg^T] with a damped BFGS
                 # inverse-Hessian estimate built from the (s, y) pairs the
                 # optimisation already produces. Everything else - the SE(3)
                 # tangent, the trust region, the stopping rule, the carry -
                 # is unchanged, so an arm differs in this ONE thing.
                 #
                 # USE target_step_norm (AUTO_LR=1) WITH THIS. P = M^-1/2 and
                 # P = B are not on the same scale: one is the inverse square
                 # root of a curvature-like matrix, the other approximates
                 # H^-1 directly. A shared lr would compare step sizes rather
                 # than methods. Normalising takes DIRECTION from the
                 # estimator, which is the contribution, and magnitude from
                 # Adam's convention, which is not.
                 bfgs: bool = False,
                 # Bound on cond(B). 20 matches the effective conditioning
                 # tau=0.05 imposes on M, so both arms are damped comparably.
                 bfgs_max_cond: float = 20.0,
                 # THE CURVATURE SCREEN, AS A COSINE. The test is
                 #
                 #     y^T s / (|y| |s|) > bfgs_curv_eps
                 #
                 # so this is a real angle threshold, not an epsilon. 1e-8 was
                 # the first value and it is NOT a threshold: it accepts every
                 # pair with any positive curvature at all, including near-flat
                 # ones where rho = 1/(y^T s) is enormous and injects a huge
                 # rank-1 rho s s^T into B. Measured consequence: B grew ~10x
                 # within a frame and max |d| sat exactly on the trust-region
                 # cap (4.90e-02 = 10 * 0.002 * sqrt(6)), with the trust region
                 # as the only thing holding it.
                 #
                 # 0.01 keeps pairs whose gradient response has a real
                 # component along the step and drops the ones carrying mostly
                 # noise. Watch pairs_used/pairs_seen: if it collapses, the
                 # threshold is too strict for this loss, and that is a fact
                 # about the loss worth knowing.
                 bfgs_curv_eps: float = 1e-2,
                 # B_0 AS A DIAGONAL, NOT THE IDENTITY. rho (translation) and
                 # theta (rotation) do not share a scale, and B_0 = I asserts
                 # they do - so the first steps of every frame are wrong in a
                 # way BFGS then has to spend pairs undoing. The measured lr
                 # sweep at the top of this file is exactly that statement:
                 # dATE/dlr_rot and dATE/dlr_trans have OPPOSITE signs.
                 #
                 # Seed instead from the config's OWN tuned Adam lrs, as
                 # (lr_trans*I3, lr_rot*I3) normalised to unit mean. BFGS then
                 # starts from a known-good diagonal metric and spends its
                 # pairs learning the off-diagonal structure a diagonal cannot
                 # represent - which is the only thing it can add.
                 bfgs_b0: tuple | None = None,
                 # AUTOSCALE B AT FRAME START, instead of normalising every
                 # step.
                 #
                 # THE ERROR THIS REPLACES. Running BFGS with target_step_norm
                 # makes every step exactly `target` long - measured as
                 # mean |d| == max |d| == 1.00x Adam, to three digits, over
                 # 2490 steps. That is fine for M, whose step M^-1/2 m is
                 # scale-free anyway, and FATAL for BFGS: -B g with B ~ H^-1 is
                 # a Newton step that SHRINKS as the gradient shrinks and lands
                 # on the optimum. Fixing its length removes exactly the
                 # property BFGS exists to provide, and the optimiser orbits
                 # instead of settling.
                 #
                 # But B_0's scale is arbitrary (the diagonal seed is
                 # normalised to unit mean), so -B_0 g is ~|g| - enormous on a
                 # loss summed over ~300k pixels. So: scale B ONCE per frame so
                 # the FIRST step is Adam-magnitude, then leave it alone. The
                 # secant equation B y = s carries the units of s/y, so
                 # accepted pairs take the scale over from there and the step
                 # decays on approach as it should.
                 bfgs_autoscale: bool = True,
                 # Hard bound on how far B may grow WITHIN a frame, relative
                 # to the scale autoscale set at its first step.
                 #
                 # clamp_spectrum bounds cond(B) but NOT its magnitude - it
                 # clamps eigenvalues to [hi/max_cond, hi] where hi is B's own
                 # largest, so the whole spectrum is free to drift upward. The
                 # trust region then catches the resulting step, but a step
                 # sitting on the cap means the metric is being ignored and the
                 # length is coming from the clamp instead. Bounding B directly
                 # keeps the DIRECTION meaningful.
                 bfgs_max_growth: float = 4.0,
                 eps: float = 1e-12,
                 # AUTO-SCALE. When > 0 the step is normalised to this norm
                 # and `lr` is ignored: delta = -target * (P m) / |P m|.
                 #
                 # WHY THIS EXISTS. lr had to be re-tuned per dataset, which
                 # undercuts the whole no-per-block-tuning claim. It does not
                 # have to: Adam's update is m/(sqrt(v)+eps) elementwise, so its
                 # step is ~lr_i per coordinate WHATEVER the gradients do, and
                 # the 6-vector norm it would take is sqrt(3 lr_t^2 + 3 lr_r^2)
                 # - a number already present in every config, already tuned by
                 # whoever tuned the dataset.
                 #
                 # So take DIRECTION from the metric, which is the contribution,
                 # and MAGNITUDE from Adam's convention, which never was.
                 #
                 # Carried over blind, PRE_LR=0.004 gave 17.49 cm on Replica
                 # against Adam's 0.24 - the measured step was 1.6e-3 where
                 # Adam's is 3.6e-3, i.e. under-stepping, on a scene whose Adam
                 # lrs are 5:1 rot:trans against TUM's 1:1.
                 target_step_norm: float = 0.0,
                 refactor_every: int = 10,
                 # Trust region, as a multiple of Adam's lr*sqrt(n). The
                 # spectrum damping bounds P's condition number, not its scale,
                 # so nothing else stops one bad metric from producing one
                 # enormous step - and one enormous step is unrecoverable here,
                 # because the map is then built from the wrong pose.
                 max_step_mult: float = 10.0,
                 # C1. MID-FRAME FIRST-MOMENT RESTART, at within-frame
                 # iteration `restart_at`. 0 = off.
                 #
                 # WHAT IT ISOLATES. The handoff bundles TWO changes at its
                 # switch, and only one of them is "a different update rule":
                 # splatam.py asserts Adam's state is EMPTY at the handoff, for
                 # the same reason carry_frame resets m every frame - carrying
                 # a moment across a change of rule injects a wrong first step.
                 # So the 6/6 handoff arm is also, unavoidably, an arm with a
                 # mid-frame momentum restart, and no measurement in either
                 # ladder separates the two.
                 #
                 # This is the restart WITHOUT the rule change: zero m, keep M,
                 # keep P, keep stepping through the metric. If it recovers
                 # most of the handoff's stability then the handoff was a
                 # restart, the second CUDA-graph capture was never buying
                 # anything, and the method runs a whole frame alone.
                 #
                 # M IS DELIBERATELY KEPT. The carry argument says M is a
                 # property of the scene geometry and m a property of the
                 # current trajectory; a restart is a statement about the
                 # trajectory only. Zeroing M as well would be a different
                 # experiment (and one m0_iso already covers).
                 restart_at: int = 0,
                 # WHAT THE RESTART DOES TO M. "off" clears the first moment
                 # only; "trace" also re-anchors M's SCALE to the current
                 # gradient while keeping its SHAPE; "iso" discards the shape
                 # as well, as the contrast that says whether the carried
                 # coupling was worth anything at that point in the frame.
                 # Default "off" so the m-only measurement stays reproducible.
                 restart_m: str = "off",
                 # PRINT THE STEP PROFILE EVERY N FRAMES AND RESET IT, so each
                 # report describes a WINDOW rather than the whole run.
                 #
                 # WITHOUT THIS A FULL-LENGTH RUN CANNOT SHOW A BREAKDOWN. The
                 # arm under investigation is healthy for ~200 frames and then
                 # diverges; a cumulative profile averages the interesting 20
                 # frames into 590 quiet ones and reports nothing unusual. The
                 # whole reason for going to full length is to SEE the window
                 # where the distribution changes, and that is invisible in a
                 # whole-run average.
                 #
                 # PURE INSTRUMENTATION: it reads accumulators and zeroes them,
                 # and touches nothing the step depends on, so it cannot change
                 # a trajectory. That is why it deliberately carries no
                 # run_name tag - the one exception to the tagging rule in this
                 # file, and safe only because it is read-only with respect to
                 # the optimiser.
                 profile_every: int = 0,
                 # OPT-IN raw per-frame series in summary() (IT/FRAME SERIES,
                 # REL-GRAD BEST/FINAL SERIES, K SERIES). profiling/lr_proxy_replay.py
                 # needs these lines in the log; every other reader just wants
                 # the p10/p50/p90 distribution above them, which always
                 # prints regardless of this flag. Off by default - a
                 # 590-frame run's raw series is several KB nobody but that
                 # one tool reads. lr_ladder.sh sets LR_SERIES=1 for the runs
                 # it hands to lr_proxy_replay.py.
                 log_series: bool = False,
                 # OPT-IN forensic snapshots. When false (the default), no
                 # trace buffers are allocated and step() executes the
                 # recorded path unchanged. When true, fixed-shape copies of
                 # the exact gradient, moment, metric, applied P and pre/post
                 # cap steps are retained for diagnostic_snapshot().
                 trace_steps: bool = False,
                 # THE DRIFT GUARD. Flag a frame whose mean pre-cap step
                 # |d|/ref exceeds drift_mult times a FROZEN baseline. 0 = off.
                 #
                 # WHY THE EXISTING GUARD CANNOT DO THIS. anomalous() judges
                 # |g_0| against _g0_ema, an EMA with ref_decay=0.98 - a
                 # ~50-frame horizon. The measured failure is a ~40-frame
                 # RAMP in which every frame is only ~10% worse than the last,
                 # so every frame passes the 2x band, is ACCEPTED into the
                 # average, and walks the reference up with it. The code's own
                 # comment says as much: it bounds the climb PER FRAME, and the
                 # record already documents the sequence version of the same
                 # failure for the stopping reference - "a gradually degrading
                 # run dragged the reference up behind it until the criterion
                 # was satisfiable again". Measured on the diverging run:
                 #
                 #   window      it5-9 mean   barred
                 #   250-275       0.20x       11%
                 #   275-300       0.33x       10%
                 #   300-325       0.35x       10%
                 #   325-350       0.84x        9%    <- frustum exit begins
                 #
                 # The guard never moved while the step distribution
                 # quadrupled.
                 #
                 # WHY THIS ONE CAN. It differs in BOTH the observable and
                 # the reference, and only the first is proven:
                 #
                 #   1. IT WATCHES A DIFFERENT QUANTITY. anomalous() reads
                 #      |g_0|; this reads |d|/ref. Those are not related:
                 #      M^{-1/2} m is EXACTLY scale-free in |g| (test 31
                 #      asserts an identical step ratio at 1x, 2x and 4x the
                 #      gradient), so the step ratio moves with the COHERENCE
                 #      of the gradient stream while |g_0| can sit still. A
                 #      degradation that shows up as a rising step ratio is
                 #      therefore invisible to a |g_0| guard by construction,
                 #      whatever its reference does.
                 #   2. ITS REFERENCE CANNOT DRIFT. ref = lr*sqrt(n) is a
                 #      constant, and the baseline is FROZEN after
                 #      drift_warmup frames, so cumulative drift is removed by
                 #      construction rather than bounded per frame.
                 #
                 # HONEST LIMIT: the run above shows `barred` flat while the
                 # step distribution quadrupled, but does NOT separate these
                 # two causes - |g_0| may simply not have moved. Either way the
                 # guard missed it, and this one watches the thing that did.
                 #
                 # The baseline is LEARNED, so no per-scene constant is
                 # introduced - only the multiplier, which is dimensionless.
                 # ON BY DEFAULT, and that is safe because it is a bit-exact
                 # no-op on any run without empty renders - see step(). Set
                 # False only to reproduce the old behaviour for an A/B.
                 dead_freeze: bool = True,
                 drift_mult: float = 0.0,
                 drift_warmup: int = 50,
                 # C2. THE CONTINUOUS HANDOFF: ramp the applied step from the
                 # metric's toward NORMALISED GRADIENT DESCENT at Adam's
                 # magnitude, over within-frame iterations [ramp_from,
                 # ramp_to]. ramp_to = 0 disables. from == to is a hard switch.
                 #
                 # The blend target is the expression the no-metric branch of
                 # step() already computes and the ladder already validated as
                 # the safe fallback - direction from the gradient, magnitude
                 # from Adam's convention, no dependence on |g|.
                 #
                 # THIS IS NOT PRE_SHRINK, and the distinction is the reason it
                 # is a separate knob. shrink pulls M toward diag(M): Adam's
                 # LEARNED per-coordinate scales, expressed in the tangent.
                 # This pulls the STEP toward the identity direction: no
                 # learned scale at all. The shrink sweep does not cover it.
                 #
                 # WHY IT IS WORTH A KNOB AT ALL. It reproduces the handoff's
                 # SHAPE - metric first, something blunter afterwards - with
                 # one optimiser, one state, one graph and therefore ONE
                 # CAPTURE PER FRAME, which is the entire reason removing the
                 # handoff was ever worth anything.
                 ramp_from: int = 0,
                 ramp_to: int = 0,
                 # Full-matrix acquisition followed by Adam-like diagonal
                 # refinement, using the SAME m and M. 0 disables; N makes
                 # within-frame iteration N the first diagonal step.
                 diag_after: int = 0,
                 # Data-driven alternative to diag_after. The caller validates
                 # each full proposal with the next render and calls
                 # activate_adaptive_diagonal() after rolling back the first
                 # proposal that does not reduce loss. The diagonal mode then
                 # stays latched for the rest of that frame.
                 adaptive_diag: bool = False,
                 # Consecutive non-improving full proposals required before
                 # switching. 1 preserves the original rule; larger values
                 # ignore isolated loss wobbles without another render.
                 adaptive_diag_patience: int = 1,
                 # Pay the adaptive proposal-validation readback only for the
                 # first N tracked frames, then replace the online rule with a
                 # fixed transition learned as the median observed switch.
                 # Zero preserves the original every-frame adaptive rule.
                 adaptive_diag_calibration_frames: int = 0,
                 # FLOOR ON THE LEARNED TRANSITION. The calibrated median is
                 # clamped up to this iteration before being installed, so a
                 # sequence whose full proposals start failing very early
                 # cannot end the coupled acquisition phase sooner than this.
                 # It clamps the INSTALLED value only; the observed samples are
                 # recorded untouched, so the summary still reports the true
                 # p10/med/p90 and whether the floor bound. Zero is the
                 # compatibility default and preserves the bare median.
                 adaptive_diag_min_iter: int = 0,
                 # Optional per-coordinate learning rates for the diagonal
                 # tail.  The full phase is unchanged.  This matters for
                 # MonoGS, whose Adam pose rates are [rot=0.003]*3 followed by
                 # [trans=0.001]*3: target_step_norm preserves only their
                 # combined norm, while a genuinely Adam-like diagonal tail
                 # must preserve the individual rates too.
                 diag_lr=None,
                 carry: bool = True,
                 # Keep normal cross-frame metric carry, but discard it when
                 # the new frame's first gradient trips the existing anomaly
                 # guard. Trace-matched isotropic replacement avoids coupling
                 # this arm to a zero/rank-one cold start. Off by default.
                 reset_m_on_anomaly: bool = False,
                 # OFF BY DEFAULT - a test of this file's own claim that m is
                 # trajectory state and carrying it across the constant-
                 # velocity re-init would inject a stale direction into the
                 # first steps of the next frame (see carry_frame()'s
                 # docstring). True carries m the same way carry=True already
                 # carries M. Independent of `carry`: can be set with M reset
                 # or carried, to isolate which one (if either) is doing
                 # anything.
                 carry_m: bool = False,
                 # THE THREE NUMBERS THAT DECIDE WHO RUNS THE FULL BUDGET.
                 # All were hard-coded; all are now measurable and sweepable.
                 # Defaults reproduce the previously shipped behaviour exactly.
                 anom_mult: float = _ANOM,
                 ref_decay: float = 0.98,      # ~50-frame horizon
                 barred_admit: float = _BARRED_ADMIT,
                 # ONLINE ACQUISITION-LR TUNING. See utils/online_lr_tuner.py
                 # for the full argument - this discovers a per-scene scale on
                 # the active acquisition knob from the run's OWN per-frame
                 # iteration counts, in one deployed pass, instead of requiring
                 # the offline paired probe (profiling/lr_probe.py) beforehand.
                 # target_step_norm owns the scale when positive; otherwise the
                 # scalar lr does. The two arms are INTERLEAVED on adjacent
                 # frames, so the active knob changes every frame until the
                 # tuner freezes. Off by default, so existing runs reproduce.
                 online_lr_tune: bool = False,
                 online_lr_block_frames: int = 12,
                 online_lr_max_halvings: int = 4,
                 online_lr_budget: int = 0,
                 online_lr_log_fn=None,
                 device=None, dtype=torch.float32):
        self.dim = int(dim)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.shrink = float(shrink)
        self.sample_metric = bool(sample_metric)
        # Relative margin an iteration must beat _g_best by to count as
        # progress. 0.01 = 1%: below that the gradient is wandering inside its
        # own noise band, which the measured curve puts at ~0.6 of |g_0|.
        self.stop_improve = float(stop_improve)
        # WHERE THE PLATEAU ANCHOR SITS. See the note in observe().
        #   "min" - the recorded behaviour: _g_best follows every new minimum.
        #   "sig" - the standard definition: it moves only on an improvement
        #           that actually beat the margin.
        self.stop_anchor = str(stop_anchor)
        self.m0_iso = bool(m0_iso)
        self.bfgs = bool(bfgs)
        self.bfgs_max_cond = float(bfgs_max_cond)
        self.bfgs_curv_eps = float(bfgs_curv_eps)
        self.bfgs_b0 = bfgs_b0
        self.bfgs_autoscale = bool(bfgs_autoscale)
        self.bfgs_max_growth = float(bfgs_max_growth)
        self.bfgs_resets = 0
        # Steps that actually received an M_inst. Host-side, which is correct
        # here ONLY because sample mode refuses both CUDA graphs - see the
        # capture note on step().
        self.sampled = 0
        # IS M A COLD EMA RIGHT NOW? Adam's 1/(1-beta^t) assumes the moment
        # started at zero. Under the carry it did not, which is why _bias1's
        # docstring says M gets no correction - correct for the case it was
        # reasoning about, and exactly wrong for the other one. With carry OFF
        # M is zeroed every frame, so at iteration 1 it is 0.05x its true size
        # and P comes out sqrt(20) ~ 4.5x too large along every direction.
        # Measured: step |d| = 1.51x Adam with carry off against 0.20-0.28x
        # with it on, pinned to the trust region, pose out of the frustum,
        # renders empty (observed max 9516 against a normal ~800k).
        #
        # AND TRACE MATCHING MAKES BOTH ARMS WRONG IDENTICALLY - the sample
        # metric is scaled to |g|^2, the same trace as g g^T - so a carry-off
        # comparison measures this bias, not the estimator.
        # TWO DISTINCT FACTS, and conflating them cost a silent no-op:
        #   _M_zeroed - M was zeroed at this frame's start. Decides SEEDING.
        #   _M_cold   - the EMA is genuinely cold. Decides BIAS CORRECTION.
        # They differ exactly when m0_iso is on: M was zeroed, but the seed
        # immediately restores the correct magnitude, so no bias correction is
        # due. Guarding the seed on _M_cold made the seed unreachable and the
        # trace came back at |g|^2/20 - group 16 catches it.
        self._M_zeroed = True
        self._M_cold = True
        self._frame_iters = 0
        # SET BY THE FRONTEND, ONCE PER FRAME, via note_frame_outcome() - see
        # that method. Consumed and reset to None in carry_frame(), the same
        # boundary that reads _frame_iters, so a frame that never calls it
        # reports None (no cap/error information) exactly like today.
        self._frame_at_cap = None
        self._frame_pose_err = None
        # Is there a usable P for THIS frame? Not "has one ever been built" -
        # see the no-metric branch in step(). False whenever M was just zeroed,
        # true again as soon as refactor() has rebuilt P from real data.
        self._have_metric = False
        # Is there a usable (s, y) pair? False at every frame start: the
        # constant-velocity prediction moves the pose against a NEW image, so
        # the previous frame's last gradient is stale and differencing across
        # that boundary would manufacture a pair from two different objectives.
        # Same reasoning that resets m in carry_frame.
        self._have_pair = False
        self.pairs_seen = 0
        self.eps = float(eps)
        self.refactor_every = int(refactor_every)
        # Keep both possible acquisition-scale bases.  MonoGS drives the
        # fixed-norm path through target_step_norm; SplaTAM's shipped configs
        # deliberately leave that at zero and tune the ordinary scalar lr.
        # The online tuner must scale the active knob, not silently multiply a
        # zero target (which used to make an opt-in SplaTAM tuner log decisions
        # while changing no optimisation step at all).
        self._base_lr = self.lr
        self.target_step_norm = float(target_step_norm)
        # Base value recorded before any online scaling.  The tuner's k always
        # multiplies a fixed reference, never the previously scaled value, so
        # repeated halvings cannot accidentally compound twice.
        self._base_target_step_norm = self.target_step_norm
        self.online_lr_tuner = None
        if online_lr_tune:
            from utils.online_lr_tuner import OnlineLRTuner
            self.online_lr_tuner = OnlineLRTuner(
                block_frames=online_lr_block_frames,
                max_halvings=online_lr_max_halvings,
                budget=online_lr_budget or None,
                log_fn=online_lr_log_fn,
            )
        self.max_step_mult = float(max_step_mult)
        self.restart_at = int(restart_at)
        self.restart_m = str(restart_m)
        self.profile_every = int(profile_every)
        self.log_series = bool(log_series)
        self.trace_steps = bool(trace_steps)
        self.dead_freeze = bool(dead_freeze)
        self.drift_mult = float(drift_mult)
        self.drift_warmup = int(drift_warmup)
        # First frame of the window the profile accumulators currently cover.
        self._profile_from = 0
        if self.restart_m not in ("off", "trace", "iso"):
            raise ValueError(
                f"restart_m must be off|trace|iso, got {self.restart_m!r}")
        self.ramp_from = int(ramp_from)
        self.ramp_to = int(ramp_to)
        if self.ramp_to and self.ramp_to < self.ramp_from:
            raise ValueError(
                f"ramp_to ({self.ramp_to}) < ramp_from ({self.ramp_from}): the "
                "ramp would run backwards. Use ramp_from == ramp_to for a hard "
                "switch.")
        self.diag_after = int(diag_after)
        self.adaptive_diag = bool(adaptive_diag)
        self.adaptive_diag_patience = int(adaptive_diag_patience)
        self.adaptive_diag_calibration_frames = int(
            adaptive_diag_calibration_frames)
        self.adaptive_diag_min_iter = int(adaptive_diag_min_iter)
        if self.adaptive_diag_min_iter < 0:
            raise ValueError("adaptive_diag_min_iter must be >= 0")
        if (self.adaptive_diag_min_iter > 0
                and self.adaptive_diag_calibration_frames <= 0):
            raise ValueError(
                "adaptive_diag_min_iter floors the CALIBRATED transition and "
                "requires adaptive_diag_calibration_frames > 0; for a fixed "
                "schedule set diag_after directly")
        if self.adaptive_diag_patience < 1:
            raise ValueError("adaptive_diag_patience must be >= 1")
        if self.adaptive_diag_calibration_frames < 0:
            raise ValueError(
                "adaptive_diag_calibration_frames must be >= 0")
        if self.adaptive_diag_calibration_frames > 0 and not self.adaptive_diag:
            raise ValueError(
                "adaptive_diag_calibration_frames requires adaptive_diag=True")
        if self.diag_after < 0:
            raise ValueError("diag_after must be >= 0")
        if self.diag_after > 0 and self.adaptive_diag:
            raise ValueError(
                "diag_after and adaptive_diag are alternative tail rules")
        if (self.diag_after > 0 or self.adaptive_diag) and self.ramp_to > 0:
            raise ValueError(
                "diagonal refinement and ramp_to are alternative tail rules; enable "
                "only one so the experiment has one interpretation")
        if (self.diag_after > 0 or self.adaptive_diag) and self.bfgs:
            raise ValueError(
                "diagonal refinement applies to the second-moment metric, not BFGS")
        # REQUESTED vs HAPPENED, the distinction this file keeps having to
        # re-learn. Both knobs fire on a within-frame iteration index, so a
        # frame that STOPS before that index never reaches them - and a run
        # whose frames all stop at 25 with restart_at=30 would report as a
        # restart arm while being bit-identical to the control. These are what
        # summary() prints to make that visible instead of silent.
        self.restarts = 0
        self.carry = bool(carry)
        self.reset_m_on_anomaly = bool(reset_m_on_anomaly)
        self.carry_m = bool(carry_m)
        self.anom_mult = float(anom_mult)
        self.ref_decay = float(ref_decay)
        self.barred_admit = float(barred_admit)

        kw = dict(device=device, dtype=dtype)
        self.m = torch.zeros(self.dim, **kw)
        self.M = torch.zeros(self.dim, self.dim, **kw)
        self._identity = torch.eye(self.dim, **kw)
        # Device-side so graph capture/replay records every actual reset.
        self._anom_m_resets_dev = torch.zeros((), **kw)
        self._custom_diag_lr = diag_lr is not None
        if diag_lr is None:
            self.diag_lr = torch.full((self.dim,), self.lr, **kw)
        else:
            if self.diag_after <= 0 and not self.adaptive_diag:
                raise ValueError(
                    "diag_lr requires diag_after > 0 or adaptive_diag=True")
            _dlr = torch.as_tensor(diag_lr, **kw).reshape(-1)
            if _dlr.numel() != self.dim:
                raise ValueError(
                    f"diag_lr must contain {self.dim} rates, got {_dlr.numel()}")
            if not bool(torch.isfinite(_dlr).all()) or bool((_dlr < 0).any()):
                raise ValueError("diag_lr entries must be finite and >= 0")
            self.diag_lr = _dlr.clone()
        # BFGS state. B approximates H^-1 and IS the preconditioner - unlike M,
        # which is inverted. _prev_g / _prev_d hold the previous iteration's
        # gradient and applied step, which is all a secant pair needs.
        # The B_0 seed, built once. Normalised to unit mean eigenvalue so it
        # sets the SHAPE of the starting metric and leaves the magnitude to
        # target_step_norm - otherwise the seed would silently be a second lr.
        if bfgs_b0 is not None:
            _t3, _r3 = float(bfgs_b0[0]), float(bfgs_b0[1])
            _d = torch.tensor([_t3] * 3 + [_r3] * 3, **kw)
            _d = _d / _d.mean().clamp_min(1e-30)
            self._B0 = torch.diag(_d)
        else:
            self._B0 = torch.eye(self.dim, **kw)
        self.B = self._B0.clone()
        self._prev_g = torch.zeros(self.dim, **kw)
        self._prev_d = torch.zeros(self.dim, **kw)
        # Accepted-pair counter, on device so it advances under a replay and
        # costs no sync in the loop - read once, in summary().
        self._sy_dev = torch.zeros((), **kw)
        self._B_scale0 = torch.ones((), **kw)
        self.P = torch.eye(self.dim, **kw)
        # Cached diagonal inverse square root of the SAME M. P remains the
        # full factor at all times; a device weight selects P_diag after the
        # configured within-frame iteration. Keeping both caches means the
        # next frame can return to full mode without an extra factorisation
        # before its first step.
        self.P_diag = torch.eye(self.dim, **kw)

        self._t = 0                # steps since the first-moment reset
        # Refactor at the FIRST opportunity, not after refactor_every steps.
        # Every iteration before the first refactor runs without a metric.
        self._since_refactor = self.refactor_every
        self._frames = 0
        self._refactors = 0
        # Frames on which an adjoint transport was actually applied. A
        # config flag says what was REQUESTED; this says what happened.
        self.transported = 0

        # Gauge diagnostic, kept on device and read once at the end - a
        # per-iteration .item() would be a sync in the hot loop. See
        # note_gauge() for what it is watching and why.
        self._gauge_lo = torch.full((), float('inf'), **kw)
        self._gauge_hi = torch.zeros((), **kw)

        # Step-size calibration, on device, read once at the end.
        #
        # THE SCALE OF lr DOES NOT CARRY OVER FROM ADAM. Adam's update is
        # m/(sqrt(v)+eps) elementwise, which is bounded at ~lr PER COORDINATE
        # whatever the gradient does - step norm ~ lr*sqrt(n). The
        # preconditioned update is lr * P m, and P's eigenvalues are
        # 1/sqrt(lambda + tau*lambda_max), which is NOT bounded relative to lr:
        # along the dominant direction of a fresh M it is already ~4.5/lr at
        # tau = 0.05. Reusing Adam's lr therefore steps several times harder
        # than Adam, which on this system is the failure mode the lr sweep
        # already identified.
        #
        # This records the realised step norm so the ratio against Adam's
        # lr*sqrt(n) is measured rather than derived on paper.
        self._step_sum = torch.zeros((), **kw)
        self._step_max = torch.zeros((), **kw)
        # WANDERING RATIO: path length over net displacement, per frame.
        #
        # path = sum of step norms, net = norm of the summed steps. In the
        # tangent space and at these step sizes that is the SE(3) displacement
        # to well within the precision anything here is read at, and it costs
        # one scalar and one 6-vector per iteration with no host read.
        #
        # WHAT IT MEASURES. The acquisition step is PINNED at 1.00x the
        # reference norm, so a frame has two ways to end. Either the pose
        # arrives - it walked more or less straight there, path ~ net, and the
        # frame length is set by the DISTANCE. Or it arrives early and then
        # dithers around the optimum, adding path but no net, and the frame
        # length is set by CONVERGENCE. Those two regimes respond to a smaller
        # step in opposite directions, which is why one lr does not serve every
        # scene: halving it costs a travel-limited frame twice the iterations
        # and lets a precision-limited one settle sooner.
        #
        # Measured, k=1 -> k=0.5 on MonoGS: fr1_desk 65.1 -> 69.8 it/frame
        # (rose, travel-limited, keep the lr) against fr2_xyz 55.9 -> 50.3,
        # fr3_office 56.4 -> 52.7, office0 69.9 -> 60.0, room0 84.4 -> 66.7
        # (all fell, precision-limited, reduce it). This ratio is meant to
        # predict that split from ONE run instead of two.
        self._frame_path = torch.zeros((), **kw)
        self._frame_net = torch.zeros(dim, **kw)
        self._wander_hist: list[float] = []
        self._step_n = 0

        # THE REALISED ITERATIONS-PER-FRAME DISTRIBUTION, NOT JUST ITS MEAN.
        #
        # summary() already reports frames= and steps=, whose ratio is the
        # quantity the beta2 rule consumes: horizon 1/(1-beta2) is set against
        # the iterations a frame ACTUALLY runs. With a stopping rule that
        # ratio is the mean of a distribution, and a single beta2 fitted to
        # the mean is simultaneously wrong for both of its tails - the more so
        # the harder the stopper cuts. On TUM fr1 the three models cut 87%,
        # 32% and 29% of their caps under three different stopping rules, so
        # their spreads are not comparable even where their means are.
        #
        # HOST-SIDE AND FREE. _frame_iters is already a host int, counted in
        # refactor() which is the one per-iteration call site guaranteed to
        # run outside a capture. Recording it costs no sync, no device memory
        # and no capture-safety argument - the reason this was never worth
        # deferring to a separate probe.
        self._frame_iter_hist: list[int] = []
        # HOW CLOSE EACH FRAME GOT, aligned index-for-index with the list above
        # (both are appended only for frames that ran). The gradient norm
        # relative to the frame's own first gradient, as the LOWEST value any
        # iteration reached and as the value at the last iteration. It is the
        # convergence stand-in for frames pinned at the iteration cap, where
        # it/frame is a constant and cannot respond to the lr: at a fixed
        # budget the lr can only change how far the frame got. NaN for a frame
        # with no valid reference gradient. Read in the frame-boundary sync
        # that already exists, so it adds no host sync.
        # LOGGED ONLY - nothing decides on it yet; it has to be shown to agree
        # with the it/frame and pose-error answers already on record first.
        self._frame_rel_best: list[float] = []
        self._frame_rel_final: list[float] = []

        # C0. THE OTHER PHASE'S STEP SIZE, in this optimiser's own units.
        #
        # SEPARATE COUNTERS, NOT THE SAME ONES. Pooling the two phases would
        # make the headline 'step |d| mean' describe neither - which is exactly
        # the failure the diluted _step_n already produced once, when refactor()
        # counted iterations on which step() had not run and the arm read as
        # 0.02x Adam against a true 0.21x.
        #
        # n IS ON DEVICE, unlike _step_n. _step_n is counted in refactor(),
        # which splatam.py SKIPS during the Adam phase - so the host counter
        # that works for the metric phase is guaranteed not to advance for this
        # one. A host counter here would divide by zero or, worse, by the other
        # phase's count.
        self._astep_sum = torch.zeros((), **kw)
        self._astep_max = torch.zeros((), **kw)
        self._astep_n = torch.zeros((), **kw)

        # THE PRE-CAP STEP PROFILE. See _profile(). All device-side and
        # fixed-shape, so it records into a capture and costs no sync.
        self._raw_hist = torch.zeros(_RAW_K, **kw)
        self._band_n = torch.zeros(_BAND_K, **kw)
        self._band_sum = torch.zeros(_BAND_K, **kw)
        self._band_max = torch.zeros(_BAND_K, **kw)
        self._band_edges = torch.tensor(_BAND_EDGES, **kw)
        self._clip_n = torch.zeros((), **kw)
        # Within-frame iteration, ON DEVICE. _frame_iters is the host copy and
        # is correct, but step() is captured and a host int would be baked in
        # at capture time - the same reason _bias1_t exists.
        self._fi_dev = torch.zeros((), **kw)
        # Drift guard state. _dsum/_dn accumulate THIS frame's pre-cap
        # |d|/ref; _dref is the frozen baseline; _dref_sum/_dref_frames build
        # it during warmup. All on device: _dsum is written from inside the
        # captured step.
        self._dsum = torch.zeros((), **kw)
        self._dn = torch.zeros((), **kw)
        self._dref = torch.zeros((), **kw)
        self._dref_sum = torch.zeros((), **kw)
        self._dref_frames = torch.zeros((), **kw)
        self._drift_n = torch.zeros((), **kw)
        # Current |g|, for the restart's trace match. See observe().
        self._gn_dev = torch.zeros((), **kw)
        # 1.0 on an iteration whose render produced a gradient, 0.0 otherwise.
        # Starts live so nothing is suppressed before the first observe().
        self._live_dev = torch.ones((), **kw)
        # Iterations skipped because the render was empty. Reported.
        self._frozen_n = torch.zeros((), **kw)
        if self.trace_steps:
            self._trace_g = torch.zeros(self.dim, **kw)
            self._trace_mhat = torch.zeros(self.dim, **kw)
            self._trace_M = torch.zeros(self.dim, self.dim, **kw)
            self._trace_P = torch.zeros(self.dim, self.dim, **kw)
            self._trace_raw = torch.zeros(self.dim, **kw)
            self._trace_applied = torch.zeros(self.dim, **kw)

        # C2. The blend weight for THIS iteration, on device so the captured
        # step reads the live value rather than the one baked in at capture
        # time. Written in place from refactor(), which is the only
        # per-iteration call site guaranteed to run outside a capture.
        self._ramp_w = torch.zeros((), **kw)
        # Sum of w over steps, so summary() can report the mean weight actually
        # applied rather than the schedule that was configured.
        self._ramp_sum = torch.zeros((), **kw)
        # Device-side hard switch and actual-use counter for the diagonal
        # refinement phase. Updated in refactor(), outside CUDA capture, and
        # read by captured step arithmetic without rebinding either tensor.
        self._diag_w = torch.zeros((), **kw)
        self._diag_sum = torch.zeros((), **kw)
        # The last full-phase gradient is retained so a rejected proposal can
        # be replaced by the diagonal update for the SAME gradient.  This is a
        # fixed-shape device copy and therefore does not add a host sync.
        self._adaptive_last_g = torch.zeros(self.dim, **kw)
        self.adaptive_switches = 0
        self._adaptive_switch_iters = []
        self._adaptive_calibration_samples = []
        self._adaptive_calibration_censored = 0
        self.adaptive_diag_fixed_after = 0
        # Raw median before the floor, kept so a run reports what the sequence
        # actually asked for as well as what was installed.
        self.adaptive_diag_learned_median = 0

        # CONVERGENCE SIGNAL. The step norm cannot provide one: M ~ E[gg^T]
        # scales as g^2 and m as g, so M^{-1/2} m is EXACTLY scale-free - as the
        # gradient shrinks by c, the preconditioner grows by c and the step is
        # unchanged. It keeps taking ~lr-sized steps at the optimum forever.
        # (Adam has the same property for the same reason, which is why the
        # early-stop pose criterion ships disabled with pose_eps=0.0 - it can
        # never fire. Measured: pose_eps=1e-4 against a step norm of ~1e-3 left
        # every frame running to the 200 cap.)
        #
        # M^-1 instead of M^-1/2 would be worse, not better: the step would
        # scale as 1/c and GROW on approach.
        #
        # But |g| itself does decay, and |g| / |g_0| - against the frame's own
        # first gradient - is scale-free ACROSS frames, so one threshold works
        # everywhere with no per-scene tuning. That is the property the loss
        # threshold does not have: loss_eps=0.004 stopped a hard frame after 28
        # iterations while loss_eps=1e-4 never stopped anything.
        self._g0 = torch.zeros((), **kw)
        # PLATEAU DETECTION, monotone by construction.
        #
        # WHY THE EXISTING CRITERIA CANNOT WORK, all three measured:
        #   loss_eps   consecutive |dLoss| below a threshold. The loss elbow is
        #              at iteration 5 (retuner, 15 probe frames, median 5-7) and
        #              tail_mass = 1.00 - EVERY bit of remaining pose motion
        #              happens after the loss stops improving. Where it fires
        #              between 5 and 200 is luck, not convergence.
        #   rel_grad   instantaneous |g|/|g_0| under a threshold. NON-MONOTONE:
        #              measured 1.000 0.847 1.043 0.768 0.877 0.843 0.684 0.619
        #              at it 0/1/2/5/10/20/30/40. A loose threshold fires on a
        #              noise dip; a strict one waits ~90 iterations.
        #   pose_eps   per-iteration step norm. Scale-free, never decays,
        #              provably dead - it ships as 0.0.
        #
        # _g_best only ever DECREASES, so an overshoot spike cannot reset it and
        # a noise dip cannot trigger it. The frame stops when the best gradient
        # it has achieved stops improving - which is what "converged" means when
        # the optimiser is orbiting rather than approaching.
        #
        # AND |g| IS OPTIMISER-AGNOSTIC, which is the property the handoff
        # needs. loss_eps failed across it because it is really a statement
        # about step size, and Adam's steps are ~4x the preconditioner's.
        self._g_best = torch.full((), float('inf'), **kw)
        self._stall = torch.zeros((), **kw)
        # THE FRAME-START LATCH IS ITS OWN FLAG, NOT `_g0 <= 0`.
        #
        # Overloading _g0 as the latch works only while |g_0| > 0. When the
        # pose leaves the frustum the render comes back EMPTY, |g| is exactly
        # zero, _g0 latches to zero - which still reads as "not started" - and
        # every subsequent iteration of that frame is treated as a frame start.
        # Measured on GSLAM TUM fr1 (PRE_HANDOFF=0, STOP_MIN=70, patience 20):
        # 'barred 7/12387' against 248 frames, i.e. 12139 of 32247 tracking
        # iterations (38%) rendering nothing, ATE 291.37cm. A healthy run reads
        # exactly /248, one judgment per frame, and that ratio is the cheapest
        # divergence detector in the summary - but only if the denominator
        # means what it says.
        self._started = torch.zeros((), **kw)
        # 1.0 once this frame has seen a zero gradient. Latched, not momentary:
        # one empty render means the pose is outside the frustum, and nothing
        # in the tracking loop steers it back.
        self._dead = torch.zeros((), **kw)
        self._rel = torch.ones((), **kw)

        # RUNNING REFERENCE, and why the per-frame one is not enough.
        #
        # |g|/|g_0| normalises by THIS frame's starting gradient, which makes a
        # diverged frame trivially easy to satisfy: its |g_0| is enormous, so a
        # 10-20x drop costs almost nothing and the frame quits at the floor
        # while still badly wrong. It commits that pose, the next frame starts
        # worse, and its |g_0| is larger still. Divergence makes the criterion
        # fire SOONER, which deepens the divergence - a self-reinforcing spiral,
        # observed directly: every frame stopping at exactly the STOP_MIN floor
        # of 10 while mapping crawled at 2 it/s. Raising the floor does not help
        # because it changes how fast the spiral runs, not whether it runs.
        #
        # _g0_ema is a slow average of per-frame INITIAL gradient norms, i.e.
        # what a typical frame starts from. Judging against it means a healthy
        # frame behaves exactly as before (its |g_0| ~ the average) while a
        # diverged frame must reduce |g| to the level a normal frame reaches -
        # which it cannot do quickly, so it runs to the cap and gets a chance
        # to recover.
        #
        # Anchored on INITIAL gradients, not on the |g| frames stopped at: a
        # stop-gradient reference is self-referential, and any tolerance != 1
        # makes its fixed point drift - stopping earlier every frame, or later
        # until everything hits the cap.
        self._g0_ema = torch.zeros((), **kw)
        # Device twin of _t, so the bias correction survives a capture.
        self._t_dev = torch.zeros((), **kw)
        # 1.0 when this frame started anomalously far from convergence.
        self._anom = torch.zeros((), **kw)
        # HOW MANY FRAMES THE GUARD BARRED - accumulated ON DEVICE.
        #
        # A host-side counter here would not advance during a CUDA graph
        # replay, which is exactly the failure that froze refactor() and the
        # step statistics (see the note in step()). These are read once, in
        # summary(). Without them there is NO way to tell a frame that ran the
        # full budget because the guard barred it from one that ran the full
        # budget because it never converged - and those need opposite fixes.
        self._barred_dev = torch.zeros((), **kw)
        self._judged_dev = torch.zeros((), **kw)
        # EMPTY RENDERS, on device for the same reason. _empty_dev counts
        # ITERATIONS with |g| == 0 and _dead_dev counts FRAMES that entered
        # that state, so the summary distinguishes "one frame flew off and
        # burned its budget" from "the whole run is outside the frustum".
        self._empty_dev = torch.zeros((), **kw)
        self._dead_dev = torch.zeros((), **kw)
        # Preallocated so stalled() allocates nothing on the stopping path.
        self._dead_stall = torch.full((), 1e9, **kw)
        # Built ONCE. torch.as_tensor(python_float) per step is a pageable
        # host->device copy, which CUDA graph capture forbids outright -
        # it is what made the first capture attempt fail.
        self._beta1_t = torch.full((), self.beta1, **kw)
        self._rel_ref = torch.ones((), **kw)

    # -- per-iteration ------------------------------------------------------

    def observe(self, grad: torch.Tensor, _count: bool = True):
        """Gradient bookkeeping ONLY - no metric update, no step, no delta.

        WHY THIS IS SEPARATE FROM step(). With a handoff, step() runs only for
        the preconditioner's share of a frame, so every quantity it maintains -
        _g0, the healthy-frame reference, _g_best, the stall counter - FREEZES
        the moment Adam takes over. A frame that had not already stalled by the
        handoff iteration could therefore never stop, and ran to the cap:
        observed as roughly half the frames at exactly 200/200 while the rest
        stopped at 14-21.

        The Adam phase calls this instead, so the plateau criterion keeps
        reading a live gradient across the whole frame. That is the property
        that made |g| the right signal for a two-phase tracker in the first
        place - a criterion that stops updating at the phase boundary throws it
        away.

        `grad` MUST be in the same SE(3) tangent step() uses. Mixing a
        7-parameter (q, t) norm into the same _g_best would compare two
        different quantities and the counter would be meaningless.
        """
        g = grad.reshape(self.dim)
        if self.adaptive_diag:
            self._adaptive_last_g.copy_(g)
        if _count:
            # step() has its own counter; this is the Adam-phase path.
            self._t += 1
        # Frame-relative gradient norm. _g0 is latched on the first step after
        # carry_frame; `torch.where` rather than a Python branch so this stays
        # capturable and free of a host sync.
        gn = g.norm()
        # KEPT ON DEVICE so refactor() can rescale M against the CURRENT
        # gradient without a host sync. refactor() runs outside any capture but
        # gn is produced inside one, so a host float would be baked in at
        # capture time - the same reason _bias1_t and _fi_dev exist.
        self._gn_dev.copy_(gn)
        # AN EMPTY RENDER CARRIES NO INFORMATION. Recorded here as a device
        # flag so step() can freeze the moments and suppress the step without a
        # host branch. See the note in step().
        self._live_dev.copy_((gn > 0).to(self._live_dev.dtype))
        _first = self._started <= 0                 # first step of this frame
        self._started.fill_(1.0)
        self._g0.copy_(torch.where(_first, gn, self._g0))
        self._rel.copy_(gn / self._g0.clamp_min(1e-20))

        # THE EMPTY RENDER. |g| is exactly zero only when the rasteriser
        # returned nothing - the pose is outside the frustum. There is no
        # information to optimise against and momentum keeps pushing in the
        # last direction, so the frame must stop rather than spend its budget
        # travelling further out. Transition-counted (_gone AND not already
        # dead) so _dead_dev is frames, not iterations. Branch-free: every
        # operand is a device scalar, so this records into a capture.
        _gone = (gn <= 0).to(self._dead.dtype)
        self._empty_dev.add_(_gone)
        self._dead_dev.add_(_gone * (1.0 - self._dead))
        self._dead.copy_(torch.maximum(self._dead, _gone))

        # Update the running reference once per frame, on that first step.
        # The contribution is CLAMPED to 3x the current average so a single
        # diverged frame - whose |g_0| can be orders of magnitude high - cannot
        # drag the reference up behind it and re-enable the early quitting the
        # reference exists to prevent.
        # OUTLIER FRAMES ARE REJECTED FROM THE REFERENCE, not clamped into it.
        #
        # Clamping the contribution to 3x bounds how fast one frame can move
        # the average but NOT how far a sequence of them can. A gradually
        # degrading run drags the reference up behind it, a frame or two at a
        # time, until the criterion is satisfiable again and the spiral
        # resumes - observed as every frame stopping at exactly STOP_MIN=10
        # after ~340 healthy ones. The reference has to be a property of
        # HEALTHY frames, so a frame that starts anomalously high must not
        # contribute to it at all.
        # AND THE REJECTION IS ABSORBING, WHICH IS THE FAILURE THIS ARM EXISTS
        # TO MEASURE. Only an ACCEPTED frame updates the reference, so the
        # reference can never learn a level it does not already accept. A
        # transient spike recovers and a gradual ramp is tracked, but a
        # SUSTAINED step in gradient level - moving into a busier part of the
        # scene - freezes the reference permanently, and every frame after it
        # is barred and runs the full budget. Reproduced on CPU in
        # profiling/anom_ratchet.py: 99 of the following 100 frames barred,
        # reference unmoved.
        #
        # barred_admit > 0 lets a barred frame move the reference slowly, so a
        # persistent new level is absorbed over ~1/barred_admit frames while a
        # divergence spiral - whose |g_0| grows every frame instead of settling
        # at a plateau - still outruns it. 0.0 is the original behaviour.
        _seed = self._g0_ema <= 0
        _ok = gn <= self.anom_mult * self._g0_ema   # within the normal range
        _accept = self.ref_decay * self._g0_ema + (1.0 - self.ref_decay) * gn
        if self.barred_admit <= 0.0:
            # DEFAULT PATH, and it must cost nothing. barred_admit is a config
            # constant, not data, so this branch is fixed for the whole run and
            # a CUDA graph capture freezing it is correct rather than a bug.
            # Computing the admission arithmetic unconditionally added five
            # scalar kernels per tracking iteration to every run that does not
            # use it.
            _reject = self._g0_ema
        else:
            # THE ADMITTED VALUE IS CLAMPED TO THE EDGE OF THE HEALTHY BAND,
            # not taken raw. A diverged frame's |g_0| can be orders of
            # magnitude high, and admitting it raw would yank the reference up
            # in one frame - after which a recovering run finds the criterion
            # trivially satisfiable and quits every frame at the floor. That is
            # the drift failure the record already documents, re-entered
            # through this door.
            #
            # Clamping bounds the climb to a factor of
            #     1 + barred_admit * (anom_mult - 1)
            # per frame (1.02 at the defaults), so the reference can still
            # absorb a persistent new level geometrically, but no single frame
            # can move it further than the band it was just judged against.
            _admit = torch.minimum(gn, self.anom_mult * self._g0_ema)
            _reject = ((1.0 - self.barred_admit) * self._g0_ema
                       + self.barred_admit * _admit)
        _updated = torch.where(_seed, gn, torch.where(_ok, _accept, _reject))
        # A DEAD FRAME IS NOT A HEALTHY ONE. gn == 0 passes the _ok test
        # trivially (0 <= anom_mult * ema), so without this an empty render
        # would be ACCEPTED into the reference and drag it toward zero - after
        # which every real frame reads as anomalous, is barred, and runs its
        # full budget. The anomaly guard would be inverted by the exact event
        # it exists to catch.
        self._g0_ema.copy_(torch.where(_first & (gn > 0),
                                       _updated, self._g0_ema))
        self._rel_ref.copy_(gn / self._g0_ema.clamp_min(1e-20))

        # PLATEAU COUNTER. An iteration counts as PROGRESS only if it beats the
        # best gradient so far by a real margin; otherwise the stall counter
        # advances. Branch-free so the body stays capturable.
        _improved = gn < self._g_best * (1.0 - self.stop_improve)
        if self.stop_anchor == "sig":
            # THE ANCHOR MOVES ONLY ON A SIGNIFICANT IMPROVEMENT.
            #
            # This is the standard early-stopping definition (Keras
            # EarlyStopping and every reference implementation):
            #     if current < best - min_delta: best = current; wait = 0
            #     else:                          wait += 1
            # `best` is the last value that actually beat the margin, so
            # sub-margin progress ACCUMULATES against a fixed anchor until it
            # collectively clears the margin, and then resets the counter.
            #
            # `_improved` implies gn < _g_best, so a plain where() is the
            # minimum; no torch.minimum needed, and it stays capturable.
            self._g_best.copy_(torch.where(_improved, gn, self._g_best))
        else:
            # RECORDED BEHAVIOUR - AND IT IS A RATCHET THAT STOPS TOO EARLY.
            #
            # _g_best follows every new minimum, including ones that did not
            # beat the margin. So a run of sub-margin improvements never resets
            # the counter AND lowers the bar the next iteration must beat: the
            # anchor runs away and accumulated progress can never register.
            #
            # Margin 1%, gradient falling 0.9%/iteration: the standard rule
            # resets at iteration 2 (0.982 < 0.990) and the frame continues;
            # this one never resets, and the frame is cut at `patience` having
            # improved 8.6% while still converging. A cut frame commits a
            # half-converged pose, the map is built from it, and the next frame
            # starts worse - which is the shape of the divergences observed at
            # frames 230 and 300 rather than at the start of a run.
            #
            # Kept as the default because every recorded result used it; the
            # comparison is STOP_ANCHOR=sig against these numbers.
            self._g_best.copy_(torch.minimum(self._g_best, gn))
        self._stall.copy_(torch.where(_improved,
                                      torch.zeros_like(self._stall),
                                      self._stall + 1.0))

        # HARD SAFETY. A frame whose own start is anomalous relative to healthy
        # frames is not eligible for early stopping at all - it runs the full
        # budget. This is the guarantee the relative criterion cannot give on
        # its own: no matter what |g|/|reference| does, a frame that begins in
        # trouble gets every iteration available to climb out.
        _anom_new = (~_ok & ~_seed).to(self._anom.dtype)
        self._anom.copy_(torch.where(_first, _anom_new, self._anom))
        # Counted on device, on the frame's first step only. Pure tensor ops,
        # so this records into a capture and advances on every replay.
        #
        # MULTIPLY BY THE MASK rather than torch.where against a fresh zeros
        # tensor: same result, and it drops two allocations plus two kernels
        # from a path that runs on every tracking iteration of every frame.
        _fmask = _first.to(self._anom.dtype)
        self._barred_dev.add_(_anom_new * _fmask)
        self._judged_dev.add_(_fmask)

        return gn, _first

    def step(self, grad: torch.Tensor,
             M_inst: torch.Tensor | None = None,
             accumulate: bool = True,
             accumulate_m: bool | None = None) -> torch.Tensor:
        """Return the parameter delta for one iteration. Capturable.

        `M_inst` is this iteration's contribution to the metric, [dim, dim]. It
        REPLACES the rank-1 `g g^T` and nothing else - the EMA, the bias
        correction, the trust region and the stopping state are untouched. Pass
        None for the shipped behaviour. See sample_metric() for what a caller
        should put in it and why it must be trace-matched.

        CAPTURE. The body stays capturable because `M_inst` arrives already
        reduced to a fixed [dim, dim] shape. The REDUCTION is not capturable -
        it runs over N Gaussians and N changes as the map grows - so the caller
        must do it outside any captured region. splatam.py refuses the
        combination outright rather than letting a graph replay a stale metric,
        which is failure mode 3 from the graph section of the ladder.

        EVERY PERSISTENT TENSOR IS UPDATED IN PLACE. `self.x = <expr>` inside
        a captured region rebinds the attribute to a tensor allocated in the
        GRAPH'S PRIVATE POOL, whose contents are only meaningful mid-replay -
        read from the host afterwards it comes back as zeros. That is not a
        subtle degradation: the gauge diagnostic read 0.0000, the step stats
        read 0, and rel_grad() fed garbage to the stopping check, which then
        quit frames after 4 iterations. m and M were unaffected only because
        they already used mul_/add_/addr_.

        Everything here is fixed-shape arithmetic on n- and nxn-sized tensors
        with no host readback, so it records into a CUDA graph. The bias
        correction uses a HOST-side counter, so the exponent is baked in at
        capture time - see `_bias1` for why that is safe.
        """
        g = grad.reshape(self.dim)
        self._t += 1
        gn, _first = self.observe(g, _count=False)
        # anomalous() becomes known only after the first gradient of the new
        # image. Reset just the stale carried metric shape before that frame's
        # first update; refactor() rebuilds P after this proposal.
        if self.reset_m_on_anomaly:
            _anom_m_reset = (_first.to(self.M.dtype)
                             * self._anom.to(self.M.dtype)
                             * self._live_dev.to(self.M.dtype))
            _iso = (gn * gn / float(self.dim)) * self._identity
            self.M.add_(_anom_m_reset * (_iso - self.M))
            self._anom_m_resets_dev.add_(_anom_m_reset)
        else:
            _anom_m_reset = None
        # _since_refactor and _step_n are NOT incremented here. This body is
        # CUDA-graph captured, and a replay executes only the recorded kernels -
        # no Python. Counting here made both advance on the ~4 eager iterations
        # per frame and stand still for the ~115 replays: refactor() then fired
        # 237 times across a run instead of ~7000, leaving P frozen, and the
        # step-size report divided a device-accumulated sum by a 30x-too-small
        # count (5.12x Adam against a true 0.17x). They are counted in
        # refactor() instead, which the tracking loop calls once per iteration
        # from OUTSIDE any capture.

        # ISOTROPIC SEED, on the frame's first step only. self._M_cold is a
        # host-side per-frame flag, constant for the whole frame, so branching
        # on it is capture-safe; _first is a device tensor, so the seed itself
        # goes through torch.where rather than a Python branch.
        if self.m0_iso and self._M_zeroed:
            _s = gn * gn / float(self.dim)
            self.M.copy_(torch.where(
                _first,
                _s * torch.eye(self.dim, dtype=self.M.dtype,
                               device=self.M.device),
                self.M))

        if self.bfgs:
            # THE SECANT PAIR, from numbers already paid for: s is the step this
            # optimiser applied last iteration, y is how the gradient responded.
            # Screened inside bfgs_update - a pair with non-positive curvature
            # along s leaves B untouched.
            #
            # _have_pair is a HOST flag that is constant for all but the first
            # iteration of a frame, so branching on it stays capture-safe.
            if self._have_pair:
                _y = g - self._prev_g
                _sy = _y.dot(self._prev_d)
                self.B.copy_(bfgs_update(self.B, self._prev_d, _y,
                                         self.bfgs_curv_eps))
                self.pairs_seen += 1
                # Host readback, so it is only counted when a probe or a
                # summary will actually read it. Cheap here because BFGS arms
                # run with graphs off anyway.
                self._sy_dev.add_(
                    (_sy > self.bfgs_curv_eps * _y.norm()
                     * self._prev_d.norm()).to(self._sy_dev.dtype))
                # BOUND |B| EVERY ITERATION, NOT EVERY refactor_every.
                #
                # V = I - rho s y^T has norm ~1/cos(y,s), so a pair sitting at
                # the screen's boundary (cos = curv_eps = 1e-2) grows B by
                # ~1e4 in ONE update. The growth bound used to live only in
                # refactor(), which rebuilds every 10 iterations - so B could
                # compound to ~1e40 and overflow float32 to inf inside that
                # window, and the next eigh raised
                #   "failed to converge ... ill-conditioned" at frame 225,
                # killing the run.
                #
                # A norm and a scalar multiply, branch-free and sync-free, so
                # it costs nothing and cannot be outrun. The eigh-based
                # CONDITION clamp stays in refactor(), where it belongs.
                _lim = self.bfgs_max_growth * self._B_scale0
                _bn = self.B.norm().clamp_min(1e-30)
                self.B.mul_(torch.clamp(_lim / _bn, max=1.0))
            self._prev_g.copy_(g)

        # AN EMPTY RENDER MUST NOT TEACH THE METRIC ANYTHING, and the
        # unguarded EMA teaches it something false and unrecoverable.
        #
        # When the pose leaves the frustum the render returns nothing and |g|
        # is EXACTLY zero. The decay still runs, so M *= beta2 every iteration
        # with nothing added - M collapses geometrically, and P = M^{-1/2}
        # grows as beta2^{-n/2}. Measured on a diverging run with 923 empty
        # iterations: a requested step of 9859x ref, with the it0-4 band
        # averaging 152x. 0.95^{-n/2} = 9859 needs n ~ 358, which is the right
        # order for what that run accumulated.
        #
        # THAT IS THE POSITIVE FEEDBACK THAT MAKES A BRIEF EXCURSION
        # PERMANENT: frustum exit -> M collapses -> P explodes -> the first
        # gradient that DOES come back is multiplied by a metric three orders
        # of magnitude too large -> thrown straight back out. It is why these
        # runs partially recover and then die rather than settling.
        #
        # The trust region bounds the magnitude but not the DIRECTION, which
        # comes from a metric built out of nothing. So this has to be fixed
        # where the metric is, not where the step is capped.
        #
        # BIT-EXACT NO-OP ON ANY HEALTHY RUN. gn > 0 on every iteration that
        # rendered anything, so _live is 1.0 throughout and every multiply
        # below is by beta1/beta2 exactly as before. It can only alter runs
        # that were already broken - which is why it is on by default rather
        # than behind a measurement flag.
        #
        # Companion to 3d381d1, which fixed the same event for the STOPPING
        # logic ("an empty render made a frame immortal") and left the metric
        # unguarded.
        # accumulate=False: TAKE A STEP, LEARN NOTHING FROM IT.
        #
        # For a caller reusing a gradient across iterations
        # (utils/grad_reuse.py), the second step sees the SAME g as the
        # first. Folding it into m and M again over-weights that one
        # direction in both moment estimates relative to the true gradient
        # sequence - it is the same measurement counted twice, not a second
        # measurement.
        #
        # WHAT THIS DOES NOT EXPLAIN, stated so nobody rediscovers it as a
        # finding: it is NOT the cause of the +25% step |d| seen in the
        # first reuse A/B. Inflating M along g makes the step SMALLER along
        # g, not larger, so the direction is wrong for that hypothesis.
        # This is an arm to measure, not a diagnosis.
        #
        # Host-side bool, so a CUDA graph would bake in whichever branch it
        # captured. Safe here only because splatam.py refuses grad_reuse
        # and a graph together; the default True leaves every captured path
        # exactly as it was.
        _live = self._live_dev if self.dead_freeze else torch.ones_like(self._live_dev)
        if accumulate:
            self._frozen_n.add_(1.0 - _live)
        # WRITTEN AS ARITHMETIC ON _live, NOT torch.where, AND THAT IS NOT
        # STYLE. torch.where(bool_tensor, 0.95, 1.0) takes its dtype from the
        # PYTHON SCALARS and returns float32 - so `M.mul_(that)` silently
        # downcast the metric's EMA. Caught by the trace-matching check in
        # group 16, which pins tr(M_0) = |g_0|^2 to 1e-8 and started failing at
        # 1.19e-08, right at float32's epsilon. On the float32 tracker it would
        # have been invisible.
        #   1 + live*(beta - 1)  ==  beta when live, 1.0 when dead
        # and it inherits _live's dtype, which is the metric's.
        # accumulate_m SEPARATES THE TWO MOMENTS, and the separation is
        # the whole point of the freeze_m arm.
        #
        # accumulate=False freezes BOTH, and was catastrophic on SplaTAM:
        # ATE 3.5 -> 10.8, progress_rejects 0 -> 43. The mechanism has been
        # stated wrongly here twice, so what is actually MEASURED (see
        # utils/test_freeze_m.py): freezing both does NOT repeat the step
        # exactly - the second step still moves ~2.5% - and over a realistic
        # refactor_every=10 schedule it ends up taking LARGER steps in the
        # polish band than accumulating does. That is the failure; the
        # "identical step twice" story was wrong.
        #
        # Freezing m ALONE stops the duplicated gradient from inflating
        # momentum while letting M advance, so the two steps stay
        # distinct. MEASURED MOTIVATION, GSLAM/fr1_desk 198 frames: with
        # reuse confined to iterations 6-39 by a cooldown, the it40+ band
        # - where every iteration renders and nothing reuses - still reads
        # 0.06x against a baseline 0.03x, and WANDER path/net goes 16.3 ->
        # 23.7. The contamination OUTLIVES the reuse window and the extra
        # motion is oscillation, not progress. m is the only state with a
        # long enough horizon (1/(1-beta2) = 50) to carry it.
        #
        # Defaults to `accumulate`, so nothing changes unless asked.
        _acc_m = accumulate if accumulate_m is None else accumulate_m
        if _acc_m:
            self.m.mul_(1.0 + _live * (self.beta1 - 1.0)).add_(
                g, alpha=1.0 - self.beta1)
        if not accumulate:
            pass
        elif M_inst is None:
            self.M.mul_(1.0 + _live * (self.beta2 - 1.0)).addr_(
                g, g, alpha=1.0 - self.beta2)
        else:
            # THE SAMPLE PATH. Same EMA, same beta2, same everything - the only
            # difference is that this iteration contributes a full-rank
            # trace-matched G^T G instead of a rank-1 g g^T. See sample_metric.
            #
            # beta2 IS NOW ARGUABLY THE WRONG DEFAULT and that is deliberate:
            # it is held at the shipped value so the first arm changes ONE
            # thing. With ~10^4 samples per backward the EMA over iterations is
            # no longer doing the estimation work it was introduced for, and
            # PRE_BETA2 near 0 is the obvious follow-up sweep - but it is a
            # SECOND experiment, not part of this one.
            self.M.mul_(self.beta2).add_(M_inst, alpha=1.0 - self.beta2)
            self.sampled += 1

        m_hat = self.m / self._bias1_t()
        if self.trace_steps:
            self._trace_g.copy_(g)
            self._trace_mhat.copy_(m_hat)
            self._trace_M.copy_(self.M)
            self._trace_P.copy_(self.P)

        # Adam's step norm at this lr. Its update is m/(sqrt(v)+eps)
        # elementwise, bounded at ~lr per coordinate WHATEVER the gradient
        # magnitude is, so this is the natural yardstick for a step here too.
        ref = (self.target_step_norm if self.target_step_norm > 0.0
               else self.lr * math.sqrt(self.dim))

        if self.bfgs and self.bfgs_autoscale:
            # Once per frame, on the first step. `_first` is a device tensor so
            # this stays branch-free; the no-op multiply on later iterations is
            # two small kernels and no sync.
            # SCALE B FOR THE CONSUMER, WHICH IS `-lr * B m`, NOT `B m`.
            #
            # Setting |B m| = ref looks right and is off by a factor of lr:
            # the applied step is lr * B m, so it came out at lr * ref =
            # 0.002 * 4.9e-3 ~ 1e-5, i.e. 500x too small, and every step after
            # the frame's first did nothing. The arithmetic that caught it:
            # mean |d| = 2.66e-04 against ref/20 = 2.45e-04 for a 20-iteration
            # frame - the mean was ENTIRELY the one fallback step per frame,
            # with the other 19 contributing nothing. ATE 230 cm, worse than
            # Adam's 154 at the same budget.
            #
            # Want |lr * B m| = ref, so |B m| = ref/lr = sqrt(dim) - which is
            # independent of lr, as it should be: B sets the shape, lr sets the
            # magnitude.
            _bm = (self.B @ m_hat).norm().clamp_min(1e-20)
            _want = ref / max(self.lr, 1e-20)
            self.B.mul_(torch.where(_first, _want / _bm,
                                    torch.ones_like(_bm)))
            # The scale this frame started from, for bfgs_max_growth. Device
            # tensor, updated in place, so it survives a replay.
            _bn = self.B.norm()
            self._B_scale0.copy_(torch.where(_first, _bn, self._B_scale0))

        if not self._have_metric:
            # NO METRIC YET *FOR THIS FRAME*. P is the identity until the first
            # refactor, and
            # `-lr * I @ m` is raw gradient descent - unscaled, unbounded, and
            # on a loss summed over ~300k pixels the gradients are large enough
            # that ten such steps destroy the pose. Measured: step norm 22.5
            # against Adam's 0.0053, a 4258x overshoot, quaternion norm to 2915
            # by frame 5.
            #
            # With no curvature information the defensible step is a NORMALISED
            # one: right direction, Adam's magnitude, no dependence on |g|.
            #
            # THIS USED TO BE KEYED ON `self._refactors == 0`, A GLOBAL COUNTER,
            # and that is only zero at the very start of the RUN. With carry
            # OFF, carry_frame resets P to the identity at the start of EVERY
            # frame while refactor() runs AFTER step() in the loop - so the
            # first step of every frame was raw gradient descent, clamped by the
            # trust region and therefore pinned exactly at max_step_mult * ref.
            # The signature is unmistakable: max |d| = 9.80e-02 against
            # 10 * 0.004 * sqrt(6) = 9.80e-02, to three digits, where carry-on
            # runs sit at 5.2-7.9e-02 and never touch the cap. Downstream the
            # pose leaves the frustum, the render comes back empty, |g| is
            # exactly zero and the frame-start latch never sets - which is what
            # 'barred 6/253' over 20 frames was reporting.
            delta = -ref * m_hat / m_hat.norm().clamp_min(1e-20)
            raw_delta = delta
        else:
            _pm = self.P @ m_hat
            if self.target_step_norm > 0.0:
                # Direction from the metric, magnitude from Adam's convention.
                # Nothing is lost by fixing the norm: the preconditioned step is
                # already scale-free (M ~ E[gg^T] scales as g^2, m as g), so it
                # never decayed on approach anyway - see the note on why a
                # pose-delta stopping criterion cannot fire.
                delta = -self.target_step_norm * _pm / _pm.norm().clamp_min(1e-20)
            else:
                delta = -self.lr * _pm
            if self.diag_after > 0 or self.adaptive_diag:
                # A STEP blend, not an M mutation. M keeps learning all 21
                # symmetric entries so the next frame's acquisition phase
                # receives the full carried metric. At w=1 this is exactly a
                # diagonal second-moment preconditioner in the SE(3) tangent.
                _pm_diag = self.P_diag @ m_hat
                if self._custom_diag_lr:
                    # Preserve Adam's PER-COORDINATE learning-rate shape in
                    # systems such as MonoGS. target_step_norm alone keeps only
                    # the vector norm and pins every tail proposal to 1x ref -
                    # exactly the failed smoke-run signature. This expression
                    # can decay and is the damped, Adam-shaped diagonal update
                    # represented by the same m/M state, with no pose handoff.
                    _delta_diag = -self.diag_lr * _pm_diag
                elif self.target_step_norm > 0.0:
                    # Backwards-compatible auto-scale behavior for callers
                    # that did not request a per-coordinate tail.
                    _delta_diag = (-self.target_step_norm * _pm_diag
                                   / _pm_diag.norm().clamp_min(1e-20))
                else:
                    _delta_diag = -self.lr * _pm_diag
                delta = delta + self._diag_w * (_delta_diag - delta)
                self._diag_sum.add_(self._diag_w)
                if self.trace_steps:
                    self._trace_P.copy_(self.P + self._diag_w
                                        * (self.P_diag - self.P))
            if self.ramp_to > 0:
                # C2. THE CONTINUOUS HANDOFF. Blend toward normalised gradient
                # descent at Adam's magnitude - the SAME expression the
                # no-metric branch above uses, so this reuses a step shape the
                # ladder has already validated rather than inventing one.
                #
                # Written as a lerp of two STEPS, which is algebraically the
                # same as stepping through P_eff = (1-w) P + w (ref/(lr|m|)) I
                # but needs no second matrix and no rebuild: P stays the pure
                # metric, so refactor()'s cached eigendecomposition is still
                # the thing it says it is.
                #
                # `self.ramp_to > 0` is a RUN-LEVEL CONSTANT, not data, so
                # branching on it cannot be frozen wrong by a capture. The
                # per-iteration quantity is _ramp_w, and that is a device
                # tensor updated in place for exactly that reason.
                _gd = -ref * m_hat / m_hat.norm().clamp_min(1e-20)
                delta = delta + self._ramp_w * (_gd - delta)
                self._ramp_sum.add_(self._ramp_w)
            if _anom_m_reset is not None:
                # P still caches the previous frame until the post-step
                # refactor. Bypass that stale factor for this proposal.
                _gd = -ref * m_hat / m_hat.norm().clamp_min(1e-20)
                delta = delta + _anom_m_reset * (_gd - delta)
            # TRUST REGION. The damping bounds P's CONDITION NUMBER but not its
            # SCALE: P ~ 1/sqrt(lambda), so a frame whose gradients collapse
            # gives a large P, and a stale P (rebuilt only every
            # refactor_every) can be badly matched to the current gradient.
            # Clamping the step to a multiple of Adam's is the standard
            # Levenberg trust region and costs two elementwise ops - no sync,
            # no branch, capturable.
            cap = self.max_step_mult * ref
            n = delta.norm()
            raw_delta = delta
            # THE STEP PROFILE, RECORDED BEFORE THE CLIP. See _profile() for
            # why pre-cap is the only version worth having: post-cap, every
            # excursion reads as exactly the cap and the distribution that
            # caused it is gone - which is why 'max |d| = 2.45e-01 to three
            # digits' could be recognised as a constant but never explained.
            self._profile(n, ref)
            delta = delta * (cap / n).clamp_max(1.0)
            # Did the trust region actually bind? REQUESTED vs HAPPENED for the
            # cap itself: max_step_mult is reported in every summary, but
            # nothing has ever said how often it was reached.
            self._clip_n.add_((n > cap).to(self._clip_n.dtype))

        # AND TAKE NO STEP AT ALL. Freezing the moments alone would leave m
        # holding the last live gradient, so the optimiser would keep pushing
        # the pose in a stale direction for as long as the render stays empty -
        # a constant-velocity drift powered by information that no longer
        # exists. Under the OLD behaviour m decayed to zero and the step died
        # with it, so freezing m without this would be strictly worse.
        #
        # There is nothing to learn and nowhere to go: hold the pose and let
        # the next frame's constant-velocity prediction have a chance at it.
        if self.dead_freeze:
            delta = delta * self._live_dev

        if self.trace_steps:
            self._trace_raw.copy_(raw_delta)
            self._trace_applied.copy_(delta)

        d = delta.norm()
        self._step_sum.add_(d)
        self._step_max.copy_(torch.maximum(self._step_max, d))
        # Same place, same units, same capture-safety as _step_sum above.
        self._frame_path.add_(d)
        self._frame_net.add_(delta)
        if self.bfgs:
            # s for the NEXT iteration's pair. The applied step, not the
            # requested one - the trust region may have shortened it, and the
            # secant equation must describe the move that actually happened.
            self._prev_d.copy_(delta)
            self._have_pair = True
        return delta

    @torch.no_grad()
    def activate_adaptive_diagonal(self) -> torch.Tensor:
        """Latch diagonal mode and replace the rejected full proposal.

        The caller must first restore the pose that existed before the rejected
        full step. M already contains that proposal's gradient, so it is kept;
        only the trajectory-dependent first moment is restarted, matching the
        established fixed diagonal transition without choosing an iteration.
        """
        if not self.adaptive_diag:
            raise RuntimeError(
                "activate_adaptive_diagonal requires adaptive_diag=True")
        if float(self._diag_w) > 0.5:
            raise RuntimeError("adaptive diagonal mode is already active")
        if not self._have_metric:
            raise RuntimeError("adaptive diagonal switch requires a metric")

        g = self._adaptive_last_g
        self.m.copy_((1.0 - self.beta1) * g)
        self._t = 1
        self._t_dev.fill_(1.0)
        self._diag_w.fill_(1.0)
        self.adaptive_switches += 1
        self._adaptive_switch_iters.append(self._frame_iters)

        # Include every acquisition gradient in the diagonal cache. This is a
        # host-side event and may refactor immediately without graph concerns.
        self._rebuild_factor()
        m_hat = self.m / self._bias1()
        pm = self.P_diag @ m_hat
        if self._custom_diag_lr:
            delta = -self.diag_lr * pm
        elif self.target_step_norm > 0.0:
            delta = (-self.target_step_norm * pm
                     / pm.norm().clamp_min(1e-20))
        else:
            delta = -self.lr * pm
        ref = (self.target_step_norm if self.target_step_norm > 0.0
               else self.lr * math.sqrt(self.dim))
        cap = self.max_step_mult * ref
        delta = delta * (cap / delta.norm().clamp_min(1e-20)).clamp_max(1.0)
        if self.dead_freeze:
            delta = delta * self._live_dev
        self._diag_sum.add_(1.0)
        return delta

    def finish_adaptive_calibration_frame(
            self, switch_iteration: int | None,
            completed_iterations: int) -> int | None:
        """Record one calibration frame and possibly install a fixed switch.

        A frame that reaches the adaptive switch contributes that iteration.
        A frame that converges first contributes its completed iteration count:
        this is right-censored evidence that no diagonal transition was needed
        earlier. Once N samples exist, their median becomes both ``restart_at``
        and ``diag_after``. ``adaptive_diag`` is then disabled, which sends the
        caller back through its original combined, no-loss-readback iteration.

        Returns the learned transition on the frame that completes calibration,
        otherwise ``None``.
        """
        if self.adaptive_diag_calibration_frames <= 0:
            return None
        if self.adaptive_diag_fixed_after > 0:
            return None
        if not self.adaptive_diag:
            raise RuntimeError(
                "adaptive diagonal calibration ended without a fixed switch")
        _done = int(completed_iterations)
        if _done < 1:
            raise ValueError("completed_iterations must be >= 1")
        if switch_iteration is None:
            _sample = _done
            self._adaptive_calibration_censored += 1
        else:
            _sample = int(switch_iteration)
            if not 1 <= _sample <= _done:
                raise ValueError(
                    "switch_iteration must lie within the completed frame")
        self._adaptive_calibration_samples.append(_sample)
        if (len(self._adaptive_calibration_samples)
                < self.adaptive_diag_calibration_frames):
            return None

        # Earlier is the conservative tie-break for an even sample count: it
        # never keeps the expensive full phase longer than the empirical median.
        _median = max(1, int(np.median(
            np.asarray(self._adaptive_calibration_samples, dtype=np.float64))))
        self.adaptive_diag_learned_median = _median
        # THE FLOOR IS APPLIED HERE, TO THE INSTALLED VALUE ONLY. Clamping the
        # median rather than gating the online switch keeps the 24 evidence
        # frames measuring the sequence's real transition, so the summary can
        # report both the observed distribution and the fact that the floor
        # overrode it. Gating the switch instead would censor the evidence from
        # below and make the floor unfalsifiable.
        _learned = max(_median, self.adaptive_diag_min_iter)
        self.restart_at = _learned
        self.diag_after = _learned
        self.adaptive_diag_fixed_after = _learned
        self.adaptive_diag = False
        return _learned

    def diagnostic_snapshot(self) -> dict:
        """Synchronise and return the exact state used by the latest step.

        This is intentionally expensive and is only available when
        ``trace_steps=True``. It must be called outside CUDA capture.
        """
        if not self.trace_steps:
            raise RuntimeError("diagnostic_snapshot requires trace_steps=True")
        M = self._trace_M.detach().double().cpu()
        P = self._trace_P.detach().double().cpu()
        evals, evecs = torch.linalg.eigh(0.5 * (M + M.T))
        p_evals, p_evecs = torch.linalg.eigh(0.5 * (P + P.T))
        mhat = self._trace_mhat.detach().double().cpu()
        metric_coeff = evecs.T @ mhat
        applied_coeff = p_evecs.T @ mhat
        raw = self._trace_raw.detach().double().cpu()
        applied = self._trace_applied.detach().double().cpu()
        ref = (self.target_step_norm if self.target_step_norm > 0.0
               else self.lr * math.sqrt(self.dim))
        spectral_step = p_evals * applied_coeff
        return {
            "gradient": self._trace_g.detach().double().cpu().tolist(),
            "gradient_norm": float(self._trace_g.norm().detach().cpu()),
            "m_hat": mhat.tolist(),
            "M": M.tolist(),
            "P": P.tolist(),
            "eigenvalues": evals.tolist(),
            "eigenvectors": evecs.tolist(),
            "P_eigenvalues": p_evals.tolist(),
            "P_eigenvectors": p_evecs.tolist(),
            "moment_metric_eigen_coefficients": metric_coeff.tolist(),
            "moment_P_eigen_coefficients": applied_coeff.tolist(),
            "spectral_step_coefficients": spectral_step.tolist(),
            "raw_step": raw.tolist(),
            "applied_step": applied.tolist(),
            "raw_step_norm": float(raw.norm()),
            "applied_step_norm": float(applied.norm()),
            "raw_step_ratio": float(raw.norm()) / max(float(ref), 1e-30),
            # se3_exp_capturable uses xi = (rho, theta): translation first,
            # rotation second. Keep the labels aligned with the update.
            "translation_step_norm": float(applied[:3].norm()),
            "rotation_step_norm": float(applied[3:].norm()),
            "live": float(self._live_dev.detach().cpu()),
            "relative_gradient": float(self._rel.detach().cpu()),
            "best_gradient": float(self._g_best.detach().cpu()),
            "stall": float(self._stall.detach().cpu()),
        }

    def step_params(self, pairs, gauge=None) -> None:
        """Apply one preconditioned step to live parameters, in place.

        `pairs` is [(param_tensor, index_or_None), ...] in the SAME order every
        iteration - the concatenation order defines what the rows and columns
        of M mean, so reordering it between calls silently scrambles the metric.
        `index` selects the per-frame slice along the last axis for the
        (1, k, num_frames) pose layout SplaTAM and Gaussian-SLAM use; pass None
        to take the whole tensor.

        REPLACES the optimiser step rather than feeding it. Transforming
        `p.grad` and letting Adam run would be a no-op: Adam renormalises per
        coordinate, so the diagonal of any pre-applied P is erased within a few
        iterations. See the module docstring.
        """
        with torch.no_grad():
            gs = []
            for p, idx in pairs:
                g = p.grad if idx is None else p.grad[..., idx]
                gs.append(g.reshape(-1))
            delta = self.step(torch.cat(gs))

            if gauge is not None:
                # PROJECT THE NULL DIRECTION OUT OF THE STEP.
                #
                # A gauge direction has EXACTLY zero curvature, so the
                # Levenberg floor gives it the longest step the matrix allows.
                # Renormalising afterwards discards that motion - but it was
                # already paid for, out of the trust region, which then clamps
                # what is left for the DOF that matter. So the null direction
                # does not merely waste a step, it CROWDS OUT the real ones.
                #
                # Measured at PRE_LR=0.008, 40 iterations: pre-normalisation
                # gauge norm 1.006 on the run that held, 1.059 and 1.112 on the
                # runs that failed. Removing the component before the step is
                # applied is exact, costs a dot product, and leaves the same lr
                # delivering strictly more useful motion.
                u = gauge.reshape(-1)
                u = u / u.norm().clamp_min(1e-20)
                delta = delta - u * torch.dot(delta, u)

            off = 0
            for p, idx in pairs:
                v = p if idx is None else p[..., idx]
                n = v.numel()
                v.add_(delta[off:off + n].reshape(v.shape))
                off += n

    def note_gauge(self, norm: torch.Tensor) -> None:
        """Record a gauge-quantity magnitude, sync-free.

        THE FAILURE THIS EXISTS TO CATCH. A preconditioner gives LOW-curvature
        directions LONGER steps - that is the whole point of it. A direction
        with EXACTLY zero curvature therefore receives the longest step the
        Levenberg floor allows, 1/sqrt(tau*lambda_max).

        If the parameterisation contains a gauge freedom - a direction the loss
        is invariant to - that is precisely a zero-curvature direction, and the
        preconditioner will pour its largest steps into it. Nothing in the loss
        objects, because by definition nothing in the loss depends on it. The
        damage is indirect and slow: the gauge quantity drifts, and whatever it
        parameterises silently rescales.

        SplaTAM's `cam_unnorm_rots` is exactly this - an unnormalised quaternion
        that is F.normalize'd at use, so its radial direction is an exact null
        direction and its norm sets the effective rotation step size. Measured:
        tracking matched the Adam control frame for frame up to ~14, then
        degraded, which is ~1260 steps of accumulated drift.

        Diagonal Adam is immune, for a reason worth stating: it divides by
        sqrt(v_i), so a coordinate with no gradient signal gets no step. It is
        blind to the null direction rather than attracted to it.
        """
        self._gauge_lo.copy_(torch.minimum(self._gauge_lo, norm.detach()))
        self._gauge_hi.copy_(torch.maximum(self._gauge_hi, norm.detach()))

    def note_frame_outcome(self, pose_err: float | None = None,
                           at_cap: bool | None = None) -> None:
        """Report the frame that just finished, for the online lr tuner's
        cap-bound fallback (see utils/online_lr_tuner.py). Call once, after
        the frame's pose has committed and BEFORE the next call to
        carry_frame() - carry_frame() reads and resets these.

        Everything here is a plain host value the caller already has (GSLAM
        computes both per committed frame for its own logging); this adds no
        device work and no new sync.

        pose_err: the committed pose's error against ground truth (e.g. mean
        absolute translation error) - only used as a fallback signal, when a
        rung turns out to be mostly at the iteration cap and it/frame alone
        cannot answer the tuning question. Omit it (the default) for a
        deployment with no ground truth; a cap-bound rung then simply declines
        to tune, as it always has.

        at_cap: whether THIS frame ran its own full iteration budget. Needed
        explicitly wherever the budget can vary frame to frame (GSLAM's
        num_iters can double on a high-loss frame) - a caller with a FIXED
        budget can leave this out and rely on online_lr_budget instead, as
        MonoGS and SplaTAM already do.

        A no-op when no tuner is running.
        """
        if self.online_lr_tuner is None:
            return
        self._frame_pose_err = pose_err
        self._frame_at_cap = at_cap

    def _profile(self, n: torch.Tensor, ref: torch.Tensor | float) -> None:
        """Record one PRE-CAP step norm, as a multiple of `ref`. Capturable.

        TWO QUESTIONS, ONE PASS, AND NEITHER NEEDS A SWEEP.

        1. WHAT WOULD ANY CAP HAVE DONE? The histogram is over |d_raw|/ref in
           log2 buckets, so the share of steps a trust region at ANY multiple
           would have clipped is read off the tail of one run. One run replaces
           one run per candidate cap, and unlike a sweep it also says what the
           clipped steps would have been rather than only that they were
           clipped.

        2. WHERE IN THE FRAME DO THE EXCURSIONS HAPPEN? The bands split by
           within-frame iteration. If the metric's spikes sit at iterations 0-5
           they are acquisition transients and a handoff-free tail is quiet
           anyway; if they are spread across all 20 then a handoff-free run
           keeps taking them near convergence, where a bad step commits a bad
           pose and the map is built from it.

        RECORDED PRE-CAP, WHICH IS THE WHOLE POINT. After the clip every
        excursion reads as exactly `cap` and the distribution that produced it
        is unrecoverable - the reason 'max |d| = 2.45e-01, to three digits,
        across configs' could be identified as a constant and never explained.
        _step_sum/_step_max stay POST-cap so every number already on record
        keeps its meaning; this is a second, separate set.

        THE BANDS ARE A DIAGNOSTIC, NOT A POLICY. Their edges are fixed
        iteration counts, which is exactly the per-scene constant that must not
        appear in a FIX - a scene with 40-iteration frames and one with 200 do
        not share them. That is fine for a histogram whose job is to describe
        one run, and it is why the bound this is measuring against is `ref`,
        which carries no such constant.
        """
        r = (n / ref).clamp_min(1e-6)
        # THE DRIFT GUARD FEEDS OFF THE PRE-CAP RATIO, and that is not
        # incidental. With PRE_MAX_STEP=1.0 the applied step is clipped at
        # exactly 1x ref, so a post-cap signal would saturate and the ramp -
        # the thing worth detecting - would be invisible precisely in the
        # configuration that needs it.
        self._dsum.add_(r)
        self._dn.add_(1.0)
        # log2 buckets, _RAW_LO..: bucket k covers [2^(k+_RAW_LO), 2^(k+1+...)).
        i = (torch.log2(r).floor() - _RAW_LO).clamp(0, _RAW_K - 1).long()
        self._raw_hist.scatter_add_(
            0, i.reshape(1), torch.ones(1, dtype=self._raw_hist.dtype,
                                        device=self._raw_hist.device))
        # Which iteration band. _fi_dev is maintained in refactor(), the only
        # per-iteration call outside a capture, so it advances under replays.
        # right=True, NOT the default. torch.bucketize defaults to
        # boundaries[i-1] < v <= boundaries[i], which puts an iteration sitting
        # exactly ON an edge into the band BELOW it - so iteration 5 would be
        # reported in it0-4 and every band would be shifted by its own first
        # element. right=True gives the half-open convention the band names
        # claim: boundaries[i-1] <= v < boundaries[i].
        b = torch.bucketize(self._fi_dev, self._band_edges, right=True).clamp(
            0, _BAND_K - 1).long().reshape(1)
        self._band_n.scatter_add_(0, b, torch.ones_like(self._band_n[:1]))
        self._band_sum.scatter_add_(0, b, r.reshape(1).to(self._band_sum.dtype))
        self._band_max.scatter_reduce_(
            0, b, r.reshape(1).to(self._band_max.dtype), reduce="amax")

    def note_applied_step(self, dxi: torch.Tensor) -> None:
        """C0. Record a step this optimiser did NOT take, in its own units.

        Call with the SE(3) tangent motion actually applied by whatever else
        moved the pose - tangent_of_pose_delta() computes it from the two
        poses. Capturable: fixed shapes, in-place accumulation, no sync.

        WHAT THE NUMBER IS FOR. The handoff arm holds 6/6 at 29.4 iters/frame
        where the preconditioner alone at the same budget is 1/3. Two stories
        fit that equally well and they call for opposite fixes:

          DIRECTION  Adam's tail steps the same distance in better places, and
                     the metric is the thing going wrong. Then shrinking it,
                     capping it and stopping later are the right candidates.
          DISTANCE   Adam's tail simply moves further, and 29 metric steps
                     cannot reach the pose no matter how good the metric is.
                     Then every one of those candidates pushes the wrong way,
                     and the knob is lr.

        The only step statistic on record - 0.21x here, 0.12x on GSLAM - is the
        metric phase's. Nothing has ever measured the other one, so nothing has
        ever separated the two stories. This is that measurement.
        """
        # Cast AFTER the norm. tangent_of_pose_delta works in float64 because
        # the differencing that produces dxi cancels away float32 entirely; by
        # this point the magnitude is an ordinary ~1e-3 number and the
        # accumulator's own dtype is ample.
        d = dxi.norm().to(self._astep_sum.dtype)
        self._astep_sum.add_(d)
        self._astep_max.copy_(torch.maximum(self._astep_max, d))
        self._astep_n.add_(1.0)

    def drifted(self) -> torch.Tensor:
        """Is THIS frame's mean step above the frozen healthy baseline?

        Companion to anomalous(), and deliberately a different measurement.
        anomalous() asks 'did this frame START badly', from |g_0| against a
        DECAYING average of |g_0|. This asks 'is this frame STEPPING badly',
        from |d|/ref against a FROZEN average of |d|/ref.

        The distinction is what the diverging run turned into a measurement:
        over the 50 frames in which the step distribution quadrupled, the
        decaying reference tracked the degradation and barred stayed at 9-11%.
        A frozen normaliser cannot do that.

        Returns 0.0 until the baseline has been established (drift_warmup
        frames), so an unwarmed run never bars anything - the guard is inert
        rather than trigger-happy while it is still learning.
        """
        if self.drift_mult <= 0.0:
            return torch.zeros((), dtype=self._dref.dtype,
                               device=self._dref.device)
        _dm = self._dsum / self._dn.clamp_min(1.0)
        return ((self._dref > 0)
                & (_dm > self.drift_mult * self._dref)).to(self._dref.dtype)

    def _ramp_weight(self, it: int) -> float:
        """Blend weight for within-frame iteration `it`. Host-side, capture-free."""
        if self.ramp_to <= 0:
            return 0.0
        if self.ramp_to == self.ramp_from:      # hard switch, the handoff shape
            return 1.0 if it >= self.ramp_from else 0.0
        return min(1.0, max(0.0, (it - self.ramp_from)
                            / float(self.ramp_to - self.ramp_from)))

    def _bias1_t(self) -> torch.Tensor:
        """Bias correction as a DEVICE tensor, so a capture cannot freeze it.

        _bias1() returns a Python float computed from a host-side counter. Under
        CUDA graph capture that value is baked into the recorded kernels, so
        every replay would apply iteration N's correction - wrong for the whole
        early part of each frame, where the correction matters most (it is 10x
        at t=1 and ~1 by t=40).

        _t_dev is incremented on device, so the correction advances with the
        replays.
        """
        # ADVANCED ONLY ON LIVE ITERATIONS. The correction answers "how many
        # gradients has m averaged"; empty renders contribute none, and letting
        # the clock run through them anneals the correction toward 1 for a
        # moment that has not actually matured.
        self._t_dev += (self._live_dev if self.dead_freeze
                        else torch.ones_like(self._live_dev))
        return (1.0 - torch.pow(self._beta1_t, self._t_dev)).clamp_min(1e-8)

    def _bias1(self) -> float:
        """Bias correction for the first moment only.

        M gets NONE. Adam's 1/(1 - beta^t) assumes the moment started at zero;
        under a warm start it did not, and applying the correction would
        inflate the metric for the first steps of every frame. m is reset every
        frame (see carry_frame), so its correction is honest.
        """
        return max(1.0 - self.beta1 ** self._t, 1e-8)

    def refactor(self, force: bool = False) -> bool:
        """Rebuild the applied preconditioner. NOT capturable - call outside.

        Returns whether it actually refactorised, so a caller can account for
        the cost.
        """
        # Called once per iteration from the tracking loop, OUTSIDE any
        # captured region - so this is the only place a host-side counter can
        # be trusted to advance on every iteration, replays included.
        self._since_refactor += 1
        self._step_n += 1
        self._frame_iters += 1
        # Device mirror for the captured step, which cannot read a host int.
        self._fi_dev.fill_(float(self._frame_iters))

        # C1 AND C2 ARE DRIVEN FROM HERE, and it has to be here. Both key on a
        # within-frame iteration index, and _frame_iters is the only such
        # counter that advances on every iteration including CUDA-graph
        # replays - because this method is the only per-iteration call the
        # tracking loop makes from OUTSIDE the capture. A counter incremented
        # inside step() stands still for the replays; that is failure mode 2 of
        # the graph section, and it cost a run that reported 237 refactors
        # where it should have had ~7000.
        #
        # OFF BY ONE, DELIBERATELY. This runs AFTER step(), so _frame_iters has
        # just become i+1 for the step i that ran. Firing on == restart_at
        # therefore makes iteration `restart_at` the FIRST one to see the fresh
        # moment - the same semantics as the handoff, where Adam's state is
        # empty at iteration `handoff` and not one step later.
        if self.restart_at > 0 and self._frame_iters == self.restart_at:
            # ZERO m AND ITS BIAS CLOCK TOGETHER. m alone would leave
            # 1/(1-beta1^t) at ~1 with t large, so the first step after the
            # restart would come out a factor (1-beta1) = 0.1 too short and the
            # arm would be measuring a stall, not a restart. carry_frame resets
            # both for the same reason.
            self.m.zero_()
            self._t = 0
            self._t_dev.zero_()
            self.restarts += 1
            if self.restart_m != "off":
                # THE OTHER MOMENT. The handoff clears Adam's exp_avg AND
                # exp_avg_sq; clearing m alone recovered 39% of the step the
                # Adam tail takes (0.10x -> 0.19x of ref at it20-39, against
                # Adam's 0.33x), and matching the EMA horizons another 17%.
                # Half the gap is M still carrying the acquisition phase's
                # large gradients, which makes P too small and the step too
                # short exactly when the frame has stopped converging.
                #
                # SCALE ONLY, SHAPE KEPT - which is the whole design. A scalar
                # multiply leaves M's eigenvectors untouched, so the coupling
                # structure the carry exists to accumulate survives intact;
                # only the magnitude, which is stale by construction during
                # convergence, is re-anchored to the CURRENT gradient. That is
                # the same trace-matching convention sample_metric already
                # uses, and it carries no iteration index and no per-scene
                # constant.
                _t2 = self._gn_dev * self._gn_dev
                if self.restart_m == "iso":
                    # THE CONTRAST ARM: discard the shape too. If iso matches
                    # trace, the carried coupling was contributing nothing at
                    # the restart point, and the method's central claim does
                    # not hold there. That is the falsifying outcome and it
                    # should be cheap to ask for.
                    self.M.copy_(torch.eye(self.dim, dtype=self.M.dtype,
                                           device=self.M.device)
                                 * (_t2 / float(self.dim)))
                else:
                    _tr = torch.diagonal(self.M).sum().clamp_min(1e-30)
                    self.M.mul_(_t2 / _tr)
                # P IS BUILT FROM M, so a rescale that does not force a rebuild
                # would not reach the step for up to refactor_every iterations -
                # i.e. for most of the tail this is trying to fix. Falling
                # through to the rebuild below is the point.
                self._since_refactor = self.refactor_every

        if self.ramp_to > 0:
            # copy_(), never assignment: _ramp_w is READ inside the captured
            # step, and rebinding it here would leave the graph replaying
            # against the tensor recorded at capture time - failure mode 3, the
            # one that froze P at its largest value for an entire run.
            self._ramp_w.fill_(self._ramp_weight(self._frame_iters))

        if self.diag_after > 0:
            # refactor() runs after step i. When the count becomes N, step N
            # must be the first diagonal one, matching PRE_RESTART and the old
            # handoff semantics. Force a rebuild at the boundary so P_diag is
            # based on every acquisition gradient rather than a stale cache.
            self._diag_w.fill_(1.0 if self._frame_iters >= self.diag_after
                               else 0.0)
            if self._frame_iters == self.diag_after:
                self._since_refactor = self.refactor_every

        if not force and self._since_refactor < self.refactor_every:
            return False
        self._rebuild_factor()
        return True

    def _rebuild_factor(self) -> None:
        """Refresh P from the current metric without advancing an iteration.

        ``refactor()`` normally owns both jobs: it advances the host counters
        for one completed optimiser update and, when due, rebuilds the cached
        factor.  A frame-boundary coordinate transport is different.  It
        changes M before the first update of the new frame, so P must change at
        the same instant, without pretending that an optimiser iteration ran.

        Keeping this as a separate internal operation also prevents the first
        update after a transport from using P in the old tangent basis while M
        is already in the new one.
        """
        # copy_(), NOT assignment. P is READ inside the CUDA-graph-captured
        # step; rebinding it here - from outside the capture - leaves the graph
        # replaying against whatever tensor existed when it was recorded. That
        # is iteration ~4, when M is nearly cold and P is therefore at its
        # LARGEST, so every replay took a near-maximal step and the trust region
        # clamped it: 5.25x Adam with max sitting exactly on the cap, against
        # 0.17x eager. The metric was frozen at its worst value for the whole
        # run and nothing raised.
        # BIAS-CORRECT ONLY A COLD EMA. See _M_cold. The factor anneals to 1
        # within ~40 iterations, so this touches the early part of a frame -
        # which is the only part where a cold M exists at all.
        if self.bfgs:
            # B IS the preconditioner - no inverse square root. It approximates
            # H^-1 directly, where M approximates the curvature and must be
            # inverted. Clamping happens here rather than in step() because
            # eigh allocates and may sync, exactly as for _inv_sqrt_damped.
            if not bool(torch.isfinite(self.B).all()):
                # REQUESTED vs HAPPENED, once more. A fallback that is not
                # counted is a run that reports as BFGS while stepping through
                # the identity.
                self.bfgs_resets += 1
            _B = clamp_spectrum(self.B, self.bfgs_max_cond)
            # Bound the MAGNITUDE too, not just the condition number.
            _lim = self.bfgs_max_growth * self._B_scale0
            _bn = _B.norm().clamp_min(1e-30)
            _B = _B * torch.clamp(_lim / _bn, max=1.0)
            self.B.copy_(_B)
            self.P.copy_(self.B)
        else:
            _M = self.M
            if self._M_cold and self._frame_iters > 0:
                _M = self.M / max(1.0 - self.beta2 ** self._frame_iters, 1e-8)
            self.P.copy_(_inv_sqrt_damped(_M, self.tau, self.eps, self.shrink))
            if self.diag_after > 0 or self.adaptive_diag:
                self.P_diag.copy_(
                    _inv_sqrt_damped_diag(_M, self.tau, self.eps))
        self._have_metric = True
        self._since_refactor = 0
        self._refactors += 1

    # -- per-frame ----------------------------------------------------------

    def carry_frame(self, transport: torch.Tensor | None = None) -> None:
        """Start a new frame.

        CARRIES M, RESETS m. The split is the point of the method:

          M is a property of the scene geometry as seen from the camera -
            which DOF are observable, how they couple, how the depth
            distribution conditions them. Adjacent frames share it. And
            because gg^T is rank-1, a cold M is SINGULAR until iteration n,
            which is the standard objection to full-matrix methods at short
            horizons. Carrying it is what removes that objection: the frame
            starts with a full-rank metric and spends its iterations USING it
            rather than ESTIMATING it.

          m is a property of the current optimisation trajectory. After the
            constant-velocity prediction re-initialises the pose against a NEW
            image, the last gradient of the previous frame is stale, and
            carrying it would inject a wrong direction into the first steps of
            every frame.

        `transport` is the SE(3) adjoint that maps the previous frame's twist
        coordinates into the new frame's (see transport_metric). Passing None
        carries the numeric matrix unchanged, so the caller is asserting that
        the coordinate basis is unchanged or that the adjoint correction is
        intentionally disabled. Set ``carry=False`` to reset M.
        """
        # THE DRIFT GUARD'S PER-FRAME ROLL-UP, and it runs BEFORE _frames is
        # incremented so `self._frames` is still the count of COMPLETED frames
        # and _dsum/_dn still hold the frame that just ended.
        #
        # All device arithmetic, no sync: the counters are read once, in
        # summary(). A host readback here would be one sync per frame in a loop
        # this file has spent considerable effort removing syncs from.
        if self.drift_mult > 0.0 and self._frames > 0:
            _dm = self._dsum / self._dn.clamp_min(1.0)
            if self._frames <= self.drift_warmup:
                # LEARNED, NOT HARD-CODED. The baseline is this scene's own
                # healthy step level, so the only constant the guard introduces
                # is the dimensionless multiplier.
                self._dref_sum.add_(_dm)
                self._dref_frames.add_(1.0)
                if self._frames == self.drift_warmup:
                    self._dref.copy_(self._dref_sum
                                     / self._dref_frames.clamp_min(1.0))
            else:
                self._drift_n.add_(
                    ((self._dref > 0)
                     & (_dm > self.drift_mult * self._dref)).to(self._drift_n.dtype))
        self._dsum.zero_()
        self._dn.zero_()
        if self.carry_m:
            # CARRY_M TEST ARM ONLY (default False - see the constructor
            # comment). m and its bias clock move together, same reason
            # restart_at resets both together above: leaving t large while m
            # is fresh (or here, leaving t at the OLD frame's value while m is
            # the OLD frame's m) is the only way to carry it without silently
            # rescaling the first step of the new frame by a wrong bias
            # correction.
            pass
        else:
            self.m.zero_()
            self._t = 0
            self._t_dev.zero_()
        self._frames += 1
        # SNAPSHOT THE FRAME'S CONVERGENCE STATE BEFORE THE RESET BELOW WIPES
        # IT. Device-side clones and a division, no sync: they are read in the
        # single frame-boundary sync further down. _g_best is +inf and _g0 is 0
        # for a frame that never took a step, which the host side maps to NaN.
        _fin_rel = self._rel.clone()
        _fin_best = self._g_best / self._g0.clamp_min(1e-20)
        _ran = self._frame_iters > 0
        # Re-latch the reference gradient: the criterion is relative to THIS
        # frame's starting gradient, which is what makes one threshold work
        # across frames of very different difficulty.
        self._g0.zero_()
        self._rel.fill_(1.0)
        self._g_best.fill_(float('inf'))
        self._stall.zero_()
        self._started.zero_()
        self._dead.zero_()
        # Counted in refactor(), which is the only per-iteration call site
        # guaranteed to run outside a capture - same reason _step_n lives there.
        #
        # RECORDED BEFORE THE RESET, and only when the frame actually ran.
        # A zero would be a frame that never reached refactor() rather than a
        # frame that stopped instantly, and admitting it would drag every
        # percentile down for a reason that has nothing to do with stopping.
        if self._frame_iters > 0:
            self._frame_iter_hist.append(self._frame_iters)
            if self.online_lr_tuner is not None:
                # A decision made here changes NEXT frame's acquisition scale,
                # never this one's: carry_frame() runs after the previous frame
                # committed, so nothing mid-frame needs to be undone.  A
                # positive target_step_norm owns the scale; otherwise the
                # scalar lr does (the normal SplaTAM configuration).
                self.online_lr_tuner.observe(
                    self._frame_iters,
                    at_cap=self._frame_at_cap, pose_err=self._frame_pose_err)
                if self._base_target_step_norm > 0.0:
                    self.target_step_norm = self.online_lr_tuner.scaled(
                        self._base_target_step_norm)
                else:
                    self.lr = self.online_lr_tuner.scaled(self._base_lr)
        # ONE sync per FRAME, at a boundary that already syncs to commit the
        # pose - not one per iteration. Both halves are stacked so it is a
        # single 2-element transfer.
        #
        # NOT gated on _frame_iters, which is advanced by the per-iteration
        # hook rather than by step(). A caller that drives step() directly - a
        # unit test, or any loop that does not use that hook - would otherwise
        # record no ratio at all while the accumulators filled correctly. The
        # accumulators are the only thing this needs, so gate on them.
        _p, _n, _rf, _rb = torch.stack(
            [self._frame_path, self._frame_net.norm(),
             _fin_rel, _fin_best]).tolist()
        if _n > 0.0 and _p > 0.0:
            self._wander_hist.append(_p / _n)
        if _ran:
            # Same gate as _frame_iter_hist so the two series stay aligned.
            self._frame_rel_final.append(
                _rf if math.isfinite(_rf) else float('nan'))
            self._frame_rel_best.append(
                _rb if math.isfinite(_rb) else float('nan'))
        self._frame_path.zero_()
        self._frame_net.zero_()
        self._frame_iters = 0
        self._fi_dev.zero_()
        # RESET UNCONDITIONALLY, whether or not online_lr_tuner consumed them,
        # so a frame that does not call note_frame_outcome() never inherits
        # stale values from an earlier one.
        self._frame_at_cap = None
        self._frame_pose_err = None
        # C2. The ramp is PER FRAME, like the handoff it replaces - iteration
        # 20 of every frame, not iteration 20 of the run. Without this reset a
        # single frame would ramp and every later frame would start already at
        # w=1, which is a different method (normalised gradient descent with a
        # 20-iteration warmup, once) wearing this arm's tag.
        #
        # Seeded from _ramp_weight(0) rather than zeroed, so ramp_from=0 means
        # what it says instead of silently skipping its first iteration.
        if self.ramp_to > 0:
            self._ramp_w.fill_(self._ramp_weight(0))
        if self.diag_after > 0 or self.adaptive_diag:
            # Every frame reacquires with the full carried metric. copy_/fill_
            # are required because captured steps retain these tensor objects.
            self._diag_w.zero_()
        # THE PAIR NEVER CROSSES A FRAME BOUNDARY. The pose was just moved by
        # the constant-velocity prediction and the image changed, so the last
        # gradient of the previous frame and the first of this one belong to
        # different objectives - differencing them manufactures a y that
        # describes no curvature at all. B itself is carried or reset by the
        # same rule M is, below.
        self._have_pair = False
        if self.bfgs and not self.carry:
            self.B.copy_(self._B0)

        if not self.carry:
            self.M.zero_()
            self.P.copy_(torch.eye(self.dim, dtype=self.M.dtype,
                                   device=self.M.device))
            self._since_refactor = self.refactor_every  # force a rebuild
            # M starts this frame at zero, so the EMA IS cold and the bias
            # correction is now the honest one.
            # m0_iso seeds M at the correct magnitude on the first step, so
            # the EMA is NOT cold in the bias sense and 1/(1-beta2^t) would
            # over-inflate it. Without it M really does start at zero.
            self._M_zeroed = True
            self._M_cold = not self.m0_iso
            # P was just reset to the identity, so there is no metric for this
            # frame until refactor() runs - which happens AFTER the first step.
            self._have_metric = False
            return
        # Warm start: M carries the previous frame's estimate at full strength,
        # so correcting it would inflate the metric exactly as _bias1 warns.
        # Frame 1 is the exception - nothing has been carried into it yet.
        self._M_zeroed = self._frames <= 1
        self._M_cold = self._M_zeroed and not self.m0_iso

        if transport is not None:
            self.transported += 1
            self.M.copy_(transport_metric(self.M, transport.to(self.M.dtype)))
            # The metric moved; the cached factorisation no longer matches it.
            # Rebuild NOW, before the first update in the new basis. Merely
            # marking it due would rebuild only after that update, which is one
            # stale-basis step per frame and exactly where opening excursions
            # concentrate.
            self._rebuild_factor()

    @property
    def pairs_used(self) -> int:
        """Secant pairs that passed the curvature screen. One host sync."""
        return int(self._sy_dev)

    def note_objective_change(self) -> None:
        """Drop the pending secant pair - the loss being differentiated changed.

        Called when tile subsampling flips between its sparse and dense phases.
        Within a phase the mask is FIXED for the whole frame (splatam.py builds
        it once, before the iteration loop), so consecutive gradients share a
        pixel subset and y is an honest curvature measurement. Across the flip
        they do not, and the resulting pair would describe the change of
        objective rather than any curvature.

        One skipped pair per frame. Harmless for M, which averages; poisonous
        for a secant method, which multiplies.
        """
        self._have_pair = False

    # -- reporting ----------------------------------------------------------

    @property
    def frames(self) -> int:
        return self._frames

    def anomalous(self) -> torch.Tensor:
        """1.0 if this frame started far outside the healthy range.

        Such a frame is barred from early stopping entirely. The relative
        criterion cannot protect itself here: a diverged frame's |g_0| is
        enormous, so ANY ratio target is easy, and a gradually degrading
        run pulls the reference along with it until the guard dissolves.
        This is the one check divergence cannot game, because it compares
        against what healthy frames did, and diverged frames are excluded
        from that average by construction.
        """
        return self._anom

    def frame_stats(self):
        """(|g_0| this frame, running reference, anom flag) as DEVICE tensors.

        For per-frame logging by the caller. Undereferenced on purpose - the
        caller decides when to pay for the sync, and the stop check already
        reads `anomalous()` on the iterations where it looks.
        """
        return self._g0, self._g0_ema, self._anom

    def rel_grad_ref(self) -> torch.Tensor:
        """|g| / (running average of per-frame initial |g|), device tensor.

        The divergence-safe form of rel_grad(). See the constructor for why the
        per-frame reference is self-defeating on exactly the frames that matter.
        """
        return self._rel_ref

    def stalled(self) -> torch.Tensor:
        """Iterations of stalled PROGRESS - zero until the frame has improved
        on its own starting gradient at all.

        The stopping quantity for STOP_MODE=best. Monotone by construction: the
        raw counter climbs whenever no new minimum is reached and resets the
        moment one is, so an overshoot spike cannot reset it and a noise dip
        cannot trigger it.

        THE PROGRESS GATE, AND WHY IT IS NOT OPTIONAL. A frame that is DIVERGING
        does not improve either - so the raw counter climbs immediately, hits
        any patience at the STOP_MIN floor, and the frame quits while still
        wrong. That commits a bad pose, the next frame starts worse, and the
        run spirals. It is the same failure this file already records for the
        ratio criterion ("divergence makes the criterion fire SOONER... every
        frame stopping at exactly STOP_MIN"), and the patience form inherits it
        unless progress is required.

        Observed directly: with the anomaly guard ON, frames past the point of
        divergence ran 6-12 iterations each while healthy frames ran ~25, and
        keyframe selection collapsed from 18 entries to 2 within three frames.

        So the count is gated on `_g_best < |g_0| * (1 - stop_improve)`: a frame
        that has never beaten its own start reports 0 forever and runs its full
        budget. Converging frames are unaffected - they beat their start within
        an iteration or two.

        Returned as ONE tensor rather than two so the caller pays a single host
        sync per check.
        """
        _progressed = self._g_best < self._g0 * (1.0 - self.stop_improve)
        # AND THE GATE HAS AN ESCAPE HATCH, because it is also a trap. A frame
        # whose render is EMPTY has |g_0| == 0, so `_g_best < 0` is false and
        # the gate reports 0 forever: the frame can never stop, runs its full
        # budget, and momentum carries the pose further outside the frustum
        # every iteration. That is a spiral the gate creates rather than
        # prevents, and it is what turned a 130-iteration GSLAM arm into ATE
        # 291cm. A dead frame returns a count no patience can survive; the
        # caller's STOP_MIN floor still bounds how early it can act.
        return torch.where(self._dead > 0, self._dead_stall,
                           self._stall * _progressed.to(self._stall.dtype))

    def rel_grad(self) -> torch.Tensor:
        """|g| / |g_0| for the current frame, as a DEVICE tensor.

        Returned undereferenced on purpose: the caller decides when to pay for
        a sync. Checking it every iteration costs a readback per iteration; the
        early-stop machinery already batches exactly this kind of signal.
        """
        return self._rel

    def spectrum(self) -> torch.Tensor:
        """Eigenvalues of M, ascending. Diagnostic only - syncs."""
        Ms = 0.5 * (self.M + self.M.transpose(-1, -2))
        return torch.linalg.eigvalsh(Ms.double()).clamp_min(0.0)

    def condition(self) -> float:
        e = self.spectrum()
        lo, hi = float(e[0]), float(e[-1])
        return float("inf") if lo <= 0.0 else hi / lo

    def profile_due(self) -> bool:
        """Is a windowed profile report due at this frame boundary?"""
        return (self.profile_every > 0 and self._frames > 0
                and self._frames % self.profile_every == 0)

    def reset_profile(self) -> None:
        """Start a new profile window. Call AFTER printing the current one.

        ONLY the profile accumulators are cleared. summary()'s step statistics
        stay cumulative on purpose: they are the numbers already on record, and
        silently turning them into a window would make every figure in the
        ladder mean something different depending on a flag that was set for
        diagnostics.
        """
        self._raw_hist.zero_()
        self._band_n.zero_()
        self._band_sum.zero_()
        self._band_max.zero_()
        self._clip_n.zero_()
        self._profile_from = self._frames

    def step_profile(self) -> str:
        """The pre-cap step distribution, as a block. Empty if nothing ran.

        PRINTED AS A CUMULATIVE TAIL, not as raw bucket counts, because the
        question it exists to answer is 'what would a trust region at Xx have
        clipped' - and that is the tail, read directly. One run therefore
        stands in for a sweep over max_step_mult, and says what the clipped
        steps WOULD have been rather than only that they were clipped.
        """
        h = self._raw_hist.detach().cpu()
        tot = float(h.sum())
        if tot <= 0:
            return ""
        lines = []
        # Cumulative share at or above each bucket's lower edge.
        tail = 0.0
        rows = []
        for k in range(_RAW_K - 1, -1, -1):
            tail += float(h[k])
            lo = 2.0 ** (k + _RAW_LO)
            if lo >= 0.25:          # below this the tail is everything
                rows.append((lo, 100.0 * tail / tot))
        rows.reverse()
        lines.append("  would-clip at cap:  "
                     + "  ".join(f"{lo:g}x:{p:.1f}%" for lo, p in rows))
        # THE BOUND, called out. 1.0x is m^T M^-1 m <= n, which the estimator
        # satisfies by construction - so anything here is the violation rate.
        over = sum(p for lo, p in rows if lo == 1.0)
        if self.target_step_norm > 0.0:
            lines.append(f"  AT/ABOVE THE ADAM-NORM REFERENCE (1x ref): {over:.1f}% "
                         f"of steps   [cap is {self.max_step_mult:g}x]")
        else:
            lines.append(f"  ABOVE THE ESTIMATOR'S OWN BOUND (1x ref): {over:.1f}% "
                         f"of steps   [cap is {self.max_step_mult:g}x, bound is 1x]")
        n_b = self._band_n.detach().cpu()
        s_b = self._band_sum.detach().cpu()
        m_b = self._band_max.detach().cpu()
        parts = []
        for k in range(_BAND_K):
            if float(n_b[k]) <= 0:
                continue
            parts.append(f"{_BAND_NAMES[k]} mean {float(s_b[k])/float(n_b[k]):.2f}x "
                         f"max {float(m_b[k]):.2f}x n={int(n_b[k])}")
        if parts:
            lines.append("  by iteration band:  " + " | ".join(parts))
        lines.append(f"  trust region bound on {int(self._clip_n)}/{int(tot)} steps "
                     f"({100.0 * float(self._clip_n) / tot:.1f}%)")
        _w = (f" frames {self._profile_from}-{self._frames}"
              if self.profile_every > 0 else "")
        _ref_label = ("target_step_norm" if self.target_step_norm > 0.0
                      else "lr*sqrt(n)")
        return (f"STEP PROFILE (pre-cap, as multiples of ref={_ref_label})"
                f"{_w}:\n" + "\n".join(lines))

    def wander_spread(self) -> dict | None:
        """Percentiles of the per-frame path/net ratio, and the regime call.

        THRESHOLD 2.0 IS PROVISIONAL. It is the round number between "walked
        there" and "orbited", and it has NOT yet been calibrated against the
        five MonoGS and three GSLAM scenes whose answers are known. Read the
        percentiles, not the label, until that calibration exists.

        p50 rather than the mean: a handful of frames that barely move give a
        huge ratio on a tiny denominator, and the mean chases them.
        """
        if not self._wander_hist:
            return None
        a = np.asarray(self._wander_hist, dtype=np.float64)
        p10, p50, p90 = (float(x) for x in np.percentile(a, [10, 50, 90]))
        return {"n": int(a.size), "p10": p10, "p50": p50, "p90": p90,
                "mean": float(a.mean()),
                "regime": ("travel-limited (keep the lr)" if p50 < 2.0
                           else "precision-limited (try a smaller lr)")}

    def frame_iter_spread(self) -> dict | None:
        """Percentiles of the realised iterations-per-frame distribution.

        Returns None before any frame has completed. The last frame is not
        included: the count is recorded in carry_frame(), which the final
        frame is never followed by - the same boundary frames= already has,
        so `n` here matches that denominator rather than disagreeing with it
        by one.
        """
        if not self._frame_iter_hist:
            return None
        a = np.asarray(self._frame_iter_hist, dtype=np.float64)
        p10, p50, p90 = (float(x) for x in np.percentile(a, (10, 50, 90)))
        mean = float(a.mean())
        return dict(n=int(a.size), mean=mean, p10=p10, p50=p50, p90=p90,
                    lo=float(a.min()), hi=float(a.max()),
                    # THE SPREAD AS A FRACTION OF THE MEAN, which is the form
                    # the beta2 question needs: the horizon is set against the
                    # mean, so what matters is how far the tails sit from it,
                    # not their absolute width. A tight distribution makes one
                    # beta2 defensible; a wide one means the tails are running
                    # at horizons the mean never described.
                    rel=(p90 - p10) / mean if mean > 0 else float("nan"))

    def summary(self) -> str:
        lo, hi = float(self._gauge_lo), float(self._gauge_hi)
        gauge = ("" if hi == 0.0 else
                 f", gauge norm {lo:.4f}..{hi:.4f}"
                 + ("  <- DRIFTED, see note_gauge()" if (lo < 0.9 or hi > 1.1) else ""))
        # THE DISTRIBUTION BEHIND steps=/frames=. See _frame_iter_hist. This
        # sits on its own line rather than inside the comma-separated header
        # because it is six numbers, and because a reader checking whether one
        # beta2 can serve a model needs it next to beta2 rather than buried.
        _sp = self.frame_iter_spread()
        spread = ("" if _sp is None else
                  f"\n  it/frame DISTRIBUTION over {_sp['n']} frames: "
                  f"p10 {_sp['p10']:.0f}  p50 {_sp['p50']:.0f}  "
                  f"p90 {_sp['p90']:.0f}  (min {_sp['lo']:.0f}, "
                  f"max {_sp['hi']:.0f}, mean {_sp['mean']:.1f}, "
                  f"p90-p10 = {100.0 * _sp['rel']:.0f}% of mean)"
                  f"\n  horizon 1/(1-beta2) = {1.0 / max(1.0 - self.beta2, 1e-12):.0f} "
                  f"covers p10 {(1.0 / max(1.0 - self.beta2, 1e-12)) / max(_sp['p10'], 1e-9):.2f}x, "
                  f"p50 {(1.0 / max(1.0 - self.beta2, 1e-12)) / max(_sp['p50'], 1e-9):.2f}x, "
                  f"p90 {(1.0 / max(1.0 - self.beta2, 1e-12)) / max(_sp['p90'], 1e-9):.2f}x")
        _wa = self.wander_spread()
        if _wa is not None:
            spread += (
                f"\n  WANDER path/net over {_wa['n']} frames: "
                f"p10 {_wa['p10']:.2f}  p50 {_wa['p50']:.2f}  "
                f"p90 {_wa['p90']:.2f}  -> {_wa['regime']}"
                f" (p50 {'<' if _wa['p50'] < 2.0 else '>='} 2.0)")
        # THE PER-FRAME ITERATION SERIES, for the paired lr probe.
        #
        # Deciding whether a scene wants a smaller acquisition lr means
        # comparing it/frame between two runs, and the per-frame spread is wide
        # - fr1_desk reads p10/p50/p90 of 48/64/88. Comparing MEANS across two
        # runs is an unpaired test: at 50 frames the standard error on the
        # difference is ~5% of the mean, while the smallest real effect on
        # record (fr3_office) is 6.6%. That needs ~100 frames per arm.
        #
        # But frame 47 is hard in BOTH arms - same motion, same texture - so
        # differencing per frame cancels the scene difficulty and leaves only
        # what the lr changed. That is worth roughly a factor of two in frames
        # at a correlation of 0.8, which is what makes a 50-frame probe viable.
        #
        # The list is already kept host-side for the percentiles above, so this
        # costs a join. ~4 chars per frame, so a 590-frame run adds ~2.4 KB to
        # a log that is already megabytes.
        if self.log_series and self._frame_iter_hist:
            spread += (
                f"\n  IT/FRAME SERIES n={len(self._frame_iter_hist)}: "
                + ",".join(str(int(v)) for v in self._frame_iter_hist))
            # The convergence stand-in, aligned with the series above. See
            # _frame_rel_best. profiling/lr_proxy_replay.py reads these.
            spread += (
                f"\n  REL-GRAD BEST SERIES n={len(self._frame_rel_best)}: "
                + ",".join(f"{v:.4g}" for v in self._frame_rel_best)
                + f"\n  REL-GRAD FINAL SERIES n={len(self._frame_rel_final)}: "
                + ",".join(f"{v:.4g}" for v in self._frame_rel_final))
            if self.online_lr_tuner is not None:
                spread += (
                    f"\n  K SERIES n={len(self.online_lr_tuner.k_log)}: "
                    + ",".join(f"{v:g}" for v in self.online_lr_tuner.k_log))
        if self.online_lr_tuner is not None:
            spread += "\n  " + self.online_lr_tuner.summary().replace(
                "\n", "\n  ")
        # Adam's step norm at the same lr, as the yardstick: its update is
        # bounded at ~lr per coordinate, so ~lr*sqrt(n) overall.
        adam_ref = (self.target_step_norm if self.target_step_norm > 0.0
                    else self.lr * math.sqrt(self.dim))
        if self._step_n:
            mean = float(self._step_sum) / self._step_n
            steps = (f", step |d| mean {mean:.2e} max {float(self._step_max):.2e}"
                     f" = {mean / adam_ref:.2f}x Adam@lr={self.lr:g}")
        else:
            steps = ""
        # C0. THE OTHER PHASE, ON THE SAME AXIS. Printed with the RATIO of the
        # two means, because that single number is what the direction/distance
        # question turns on and computing it by hand from two lines of a log is
        # how the 0.02x-vs-0.21x confusion happened in the first place.
        _an = float(self._astep_n)
        if _an > 0:
            _amean = float(self._astep_sum) / _an
            other = (f", OTHER-PHASE |d| mean {_amean:.2e} "
                     f"max {float(self._astep_max):.2e} over {int(_an)} steps"
                     f" = {_amean / adam_ref:.2f}x Adam@lr={self.lr:g}")
            if self._step_n and self._step_sum > 0:
                other += f", ratio other/metric {_amean / mean:.2f}x"
        else:
            other = ""
        # C1 / C2. REQUESTED vs HAPPENED. Both knobs fire on a within-frame
        # iteration index, so a run whose frames all stop BEFORE that index is
        # bit-identical to the control while carrying this arm's tag. These
        # lines are what makes that visible in the log instead of in a
        # conclusion six weeks later.
        if self.drift_mult > 0.0:
            _dn = float(self._drift_n)
            _judged = max(0, self._frames - self.drift_warmup)
            drift = (f", drift-barred {int(_dn)}/{_judged} "
                     f"(mult={self.drift_mult:g}, baseline "
                     f"{float(self._dref):.3f}x ref from {self.drift_warmup} frames)")
            if self._frames > self.drift_warmup and float(self._dref) <= 0:
                drift += "  <- BASELINE NEVER FROZE: the guard is inert"
        else:
            drift = ""
        if self.restart_at > 0:
            restart = (f", restarts {self.restarts}/{self._frames} frames "
                       f"@it{self.restart_at}"
                       + (f" (M:{self.restart_m})" if self.restart_m != "off"
                          else ""))
            # GUARDED ON _frames > 0. summary() is also printed once at
            # construction, before any frame has run, where 0/0 tripped this
            # warning and told the operator their arm was the control on a run
            # that had not started yet.
            if self._frames > 0 and self.restarts < 0.5 * self._frames:
                restart += ("  <- NEVER FIRED on most frames: they stopped "
                            "before it. This arm is the control.")
        else:
            restart = ""
        if self.ramp_to > 0:
            _w = float(self._ramp_sum) / max(1, self._step_n)
            ramp = (f", ramp {self.ramp_from}->{self.ramp_to} mean w {_w:.3f}")
            if self._frames > 0 and _w < 0.02:
                ramp += ("  <- NEVER ENGAGED: frames stop before ramp_from. "
                         "This arm is the control.")
        else:
            ramp = ""
        if self.diag_after > 0:
            _dw = float(self._diag_sum) / max(1, self._step_n)
            diagonal = (f", diagonal tail @it{self.diag_after} mean w "
                        f"{_dw:.3f}")
            if self.adaptive_diag_fixed_after > 0:
                _cs = np.asarray(
                    self._adaptive_calibration_samples, dtype=np.float64)
                _cp10, _cmed, _cp90 = np.percentile(_cs, [10, 50, 90])
                diagonal += (
                    f", learned from {len(_cs)} frames "
                    f"p10/med/p90={_cp10:.0f}/{_cmed:.0f}/{_cp90:.0f} "
                    f"censored={self._adaptive_calibration_censored}"
                )
                # REQUESTED vs INSTALLED. If the floor moved the transition,
                # say so on the same line: otherwise a reader sees a median of
                # 6 beside a tail at it10 and cannot tell whether the clamp
                # bound or the calibration simply landed there.
                if (self.adaptive_diag_min_iter > 0
                        and self.adaptive_diag_learned_median > 0):
                    if (self.adaptive_diag_learned_median
                            < self.adaptive_diag_min_iter):
                        diagonal += (
                            f" FLOORED {self.adaptive_diag_learned_median}"
                            f"->{self.adaptive_diag_min_iter}")
                    else:
                        diagonal += (
                            f" floor {self.adaptive_diag_min_iter} not binding")
            if self._custom_diag_lr:
                diagonal += (f" lr {float(self.diag_lr.min()):g}.."
                             f"{float(self.diag_lr.max()):g}")
            if self._frames > 0 and _dw < 0.02:
                diagonal += ("  <- NEVER ENGAGED: frames stop before the "
                             "switch. This arm is the control.")
        else:
            if self.adaptive_diag:
                _dw = float(self._diag_sum) / max(1, self._step_n)
                if self._adaptive_switch_iters:
                    _si = np.asarray(self._adaptive_switch_iters, dtype=np.float64)
                    _p10, _med, _p90 = np.percentile(_si, [10, 50, 90])
                    _where = f", switch iteration p10/med/p90={_p10:.0f}/{_med:.0f}/{_p90:.0f}"
                else:
                    _where = ""
                diagonal = (
                    f", adaptive diagonal switches {self.adaptive_switches}/"
                    f"{self._frames} frames after "
                    f"{self.adaptive_diag_patience} consecutive failures, "
                    f"mean w {_dw:.3f}{_where}"
                )
                if self.adaptive_diag_calibration_frames > 0:
                    diagonal += (
                        f", calibration "
                        f"{len(self._adaptive_calibration_samples)}/"
                        f"{self.adaptive_diag_calibration_frames} frames"
                    )
                if self._custom_diag_lr:
                    diagonal += (f" lr {float(self.diag_lr.min()):g}.."
                                 f"{float(self.diag_lr.max()):g}")
            else:
                diagonal = ""
        # A refactor count far below steps/refactor_every means the schedule is
        # not advancing - the signature of a host counter that sits inside a
        # CUDA-graph-captured region and therefore only ticks on eager
        # iterations. It leaves P frozen, which degrades tracking silently.
        expect = self._step_n // max(1, self.refactor_every)
        stale = ("" if self._refactors >= 0.5 * expect else
                 f"  <- STALE: expected ~{expect} refactors for {self._step_n} "
                 f"steps. The schedule is not advancing; see the note in step().")
        # transport is reported because it is an ABLATION ARM: three runs whose
        # summaries were identical could not be told apart afterwards, and the
        # one that was supposed to have it on could not be confirmed.
        # BARRED FRAMES. These run the FULL iteration budget by construction,
        # so this is the share of tracking cost the guard is spending. A high
        # number with a frozen reference is the ratchet (see the note in
        # step()); a high number with a reference that tracks |g_0| means the
        # frames really are starting in trouble. The two need opposite fixes
        # and nothing else in the run distinguishes them.
        _j = float(self._judged_dev)
        _b = float(self._barred_dev)
        # EMPTY RENDERS ARE REPORTED WHENEVER THERE ARE ANY. Silence here cost
        # a full GSLAM sweep: the only visible symptom was the barred
        # denominator reading 12387 instead of 248, which requires knowing what
        # that field counts to notice at all.
        _fz = float(self._frozen_n)
        frozen = ("" if _fz <= 0 else
                  f", froze m/M on {int(_fz)} empty iterations"
                  if self.dead_freeze else
                  f", {int(_fz)} empty iterations DECAYED M (dead_freeze off)")
        _e = float(self._empty_dev)
        _d = float(self._dead_dev)
        dead = ("" if _e <= 0 else
                f", EMPTY RENDERS {int(_e)} iters over {int(_d)} frames"
                f" - pose left the frustum")
        barred = ("" if _j <= 0 else
                  f", barred {int(_b)}/{int(_j)} ({100.0 * _b / _j:.0f}%)"
                  f" [anom_mult={self.anom_mult:g}, ref_decay={self.ref_decay:g}"
                  + (f", barred_admit={self.barred_admit:g}"
                     if self.barred_admit else "") + "]")
        anom_reset = ""
        if self.reset_m_on_anomaly:
            anom_reset = (f", anomaly-M-reset="
                          f"{int(self._anom_m_resets_dev)}/{int(_j)}")
        return (f"Pose preconditioner: dim={self.dim}, "
                f"{'target|d|=%g' % self.target_step_norm if self.target_step_norm > 0 else 'lr=%g' % self.lr}, "
                f"frames={self._frames}, steps={self._step_n}, "
                f"refactors={self._refactors} (every {self.refactor_every})"
                f"{stale}, "
                f"beta2={self.beta2}, tau={self.tau}, shrink={self.shrink:g}, "
                f"carry={self.carry}{anom_reset}"
                + (", m0=iso" if self.m0_iso else "")
                # REQUESTED vs HAPPENED again. pairs_used/pairs_seen is the
                # damping rate: if most pairs are rejected the loss is too
                # noisy or too non-convex for a secant method here, and that
                # is the number that says so.
                + (f", BFGS(cond<={self.bfgs_max_cond:g}, pairs "
                   f"{self.pairs_used}/{self.pairs_seen}"
                   + (f", RESETS {self.bfgs_resets}" if self.bfgs_resets
                      else "") + ")" if self.bfgs else "")
                + ", "
                # REQUESTED vs HAPPENED, the same distinction transport draws
                # below. A silently-dropped M_inst (grad not retained, wrong
                # tensor, mapping-mode call) would leave the run looking
                # entirely plausible while measuring the shipped rank-1 method.
                # This is the line that catches it.
                + (f"samples={self.sampled}/{self._step_n} steps, "
                   if self.sample_metric else "")
                + f"transport={self.transported}/{self._frames}"
                f"{barred}{dead}{frozen}{gauge}{steps}{other}{restart}{ramp}"
                f"{diagonal}{drift}{spread}")
