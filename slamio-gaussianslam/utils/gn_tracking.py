"""
gn_tracking.py - STEP 1 of results/PLAN_second_order_tracking.md.

THE ONE QUESTION THIS ANSWERS, and nothing else:

    Does Gauss-Newton converge on an alpha-blended photometric residual?

Tracking optimises SIX numbers. SplaTAM, MonoGS, GSplatLoc and FSGS all use
Adam and pay ~100-120 iterations per frame for it, three of them bolting a
convergence heuristic on to absorb the cost. Classical direct VO solves the
same shape of problem with Gauss-Newton in 5-10. The plan argues the pose is
the one sub-problem where the exact Hessian is 6x6 and closed-form.

But the residual here is alpha-blended over anisotropic Gaussians, not a
brightness-constant image warp, and it may simply not be well-approximated
quadratically at inter-frame motion scale. That risk (P5) kills the whole
idea, so it is tested FIRST, cheaply, before a single kernel line is written.

FINITE DIFFERENCES, ON PURPOSE.

J is built with 12 extra renders per iteration (central differences over 6
pose DOF). That is crude and slow and it is the right choice here:

  - it needs NO autograd and NO kernel change, so it cannot be wrong for
    reasons of plumbing - only the maths is under test
  - it is parameterisation-agnostic, so the same probe runs on SplaTAM,
    MonoGS and Gaussian-SLAM without first porting SE(3) deltas anywhere
  - the real version computes J analytically in ONE pass, so the cost here
    says nothing about the cost there

Even at 13 renders per GN iteration against 1 per Adam iteration, 10 GN
iterations beats 120 Adam ones. But SPEED IS NOT THE MEASUREMENT. Iterations
to convergence and final pose error are.

KILL CRITERION, fixed in advance: if GN needs more than ~30 iterations, or
only converges with damping so heavy that the step direction is essentially
the gradient, STOP. That is a publishable negative and it costs a day.

WHAT IT DOES NOT DO. It does not touch the live optimisation. It renders from
poses it makes up, measures, restores nothing (it never writes), and reports.
Default off.
"""

from __future__ import annotations

import json
import math


# --------------------------------------------------------------------------
# SE(3) helpers. Left-perturbation convention: T <- exp(dxi) @ T, matching
# MonoGS's update_pose, so the prototype's parameterisation is the one the
# analytic version would inherit.
# --------------------------------------------------------------------------

def _skew(v):
    import torch
    o = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([o, -v[2], v[1]]),
        torch.stack([v[2], o, -v[0]]),
        torch.stack([-v[1], v[0], o]),
    ])


def se3_exp(xi):
    """xi = [rho(3) translation ; theta(3) rotation] -> 4x4.

    Closed form with the Taylor branch taken on a HOST float, not a device
    tensor. MonoGS's SO3_exp branches on a device value, which is both a sync
    and data-dependent control flow - a capture freezes whichever branch it
    happened to take. Irrelevant here (no capture) but there is no reason to
    copy the bug into a file that may get ported.
    """
    import torch
    rho, theta = xi[:3], xi[3:]
    ang = float(theta.norm())
    W = _skew(theta)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device)
    # SERIES BELOW 1e-2, not 1e-8. The closed forms divide (1 - cos(ang)) and
    # (ang - sin(ang)) by powers of ang, and both numerators are differences of
    # nearly equal quantities: at ang=1e-4, 1 - cos(ang) ~ 5e-9 is formed by
    # subtracting two numbers within 5e-9 of each other, costing ~9 significant
    # digits. The solver's accepted steps are ~1e-4 (measured |step| in the
    # spectrum logs), so it sits IN that band on every iteration - a cutoff of
    # 1e-8 meant the series branch was effectively never taken when it mattered.
    if ang < 1e-2:
        a2 = ang * ang
        c1 = 1.0 - a2 / 6.0 + a2 * a2 / 120.0        # sin(ang)/ang
        c2 = 0.5 - a2 / 24.0 + a2 * a2 / 720.0       # (1 - cos(ang))/ang^2
        c3 = 1.0 / 6.0 - a2 / 120.0 + a2 * a2 / 5040.0  # (ang - sin)/ang^3
    else:
        s, c = math.sin(ang), math.cos(ang)
        c1 = s / ang
        c2 = (1 - c) / ang ** 2
        c3 = (ang - s) / ang ** 3
    WW = W @ W
    R = I + c1 * W + c2 * WW
    V = I + c2 * W + c3 * WW
    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T):
    """4x4 -> xi = [rho(3); theta(3)]. The exact inverse of se3_exp.

    Needed so that a pose DIFFERENCE can be expressed in the same coordinates
    the solver steps in: d = se3_log(T_b @ inv(T_a)) is the left perturbation
    with se3_exp(d) @ T_a == T_b, so scaling it traces the geodesic from a to b.

    VALID AT EVERY ANGLE INCLUDING pi. The rotation vector is extracted through
    the QUATERNION rather than from (R - R^T), because the latter carries a
    factor ang/(2 sin ang) that blows up as ang -> pi. mat_to_quat uses
    Shepperd's method, which picks the numerically largest branch and is
    well-conditioned there.

    An earlier version asserted ang < 3.0 on the reasoning that every caller
    feeds it a small inter-frame correction. That was true only while tracking
    was working: a diverged run produces genuine ~180 degree pose errors, and
    the assert then fired thousands of times and silently discarded the
    diagnostic records for exactly the frames that most needed explaining.

    Note the V_inv coefficient is NOT singular at pi - sin(pi) = 0 and
    1 - cos(pi) = 2 give A = 1/pi^2. Only ang -> 0 needs the series branch.
    """
    import torch
    R, t = T[:3, :3], T[:3, 3]
    I = torch.eye(3, dtype=T.dtype, device=T.device)
    q = mat_to_quat(R)                       # (w, x, y, z), unit
    v = q[1:]
    vn = float(v.norm())
    # ang = 2*atan2(|v|, w) covers the full [0, pi] range without an acos of a
    # trace that numerical drift can push outside [-1, 1].
    ang = 2.0 * math.atan2(vn, float(q[0]))
    if ang > math.pi:                        # atan2 branch -> shorter rotation
        ang = 2.0 * math.pi - ang
        v = -v
    if vn < 1e-12:                           # no rotation; axis undefined
        theta = torch.zeros(3, dtype=T.dtype, device=T.device)
    else:
        theta = v / vn * ang
    W = _skew(theta)
    if ang < 1e-2:
        # SERIES, not the closed form. A = (1 - ang*sin/(2(1-cos)))/ang^2 tends
        # to 1/12, but it computes that limit as a difference of two numbers
        # both approaching 1 and then divides by ang^2 - at ang=1e-4 that
        # cancellation costs ~9 significant digits. Verified: the round trip
        # exp(log(T)) holds to 2e-16 across 1e-9..pi with this band, against
        # 4e-11 at |theta|=1e-6 before it.
        A = 1.0 / 12.0 + ang ** 2 / 720.0 + ang ** 4 / 30240.0
    else:
        A = (1.0 - ang * math.sin(ang) / (2.0 * (1.0 - math.cos(ang)))) / ang ** 2
    V_inv = I - 0.5 * W + A * (W @ W)
    return torch.cat([V_inv @ t, theta])


def se3_generators(dtype, device):
    """The six 4x4 generators G_k with d/deps_k [se3_exp(eps) @ T] = G_k @ T.

    Used to convert an autograd gradient w.r.t. a 4x4 pose into a gradient in
    the same LEFT-PERTURBATION coordinates the solver steps in. Going through a
    differentiable se3_exp instead would be wrong here: the real one reads the
    rotation angle out to a host float for its Taylor cutoff, so the
    angle-dependent coefficients become Python constants and autograd silently
    drops their contribution. That is harmless for finite differences, which
    only ever evaluate the function, and fatal for a gradient.
    """
    import torch
    G = torch.zeros((6, 4, 4), dtype=dtype, device=device)
    for k in range(3):
        G[k, k, 3] = 1.0                      # translation
    G[3, 2, 1], G[3, 1, 2] = 1.0, -1.0        # rotation about x
    G[4, 0, 2], G[4, 2, 0] = 1.0, -1.0        # about y
    G[5, 1, 0], G[5, 0, 1] = 1.0, -1.0        # about z
    return G


def mat_to_quat(R):
    """3x3 rotation -> (w, x, y, z), matching SplaTAM's build_rotation.

    Shepperd's method: pick the branch with the largest denominator, because
    the naive w-branch loses precision (and can take a sqrt of a negative)
    when w is near zero, i.e. at rotations near 180 degrees. Round-tripped
    against build_rotation in the tests rather than assumed.
    """
    import torch
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s_ = torch.sqrt(t + 1.0) * 2
        w = 0.25 * s_
        x = (m[2, 1] - m[1, 2]) / s_
        y = (m[0, 2] - m[2, 0]) / s_
        z = (m[1, 0] - m[0, 1]) / s_
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s_ = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s_
        x = 0.25 * s_
        y = (m[0, 1] + m[1, 0]) / s_
        z = (m[0, 2] + m[2, 0]) / s_
    elif m[1, 1] > m[2, 2]:
        s_ = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s_
        x = (m[0, 1] + m[1, 0]) / s_
        y = 0.25 * s_
        z = (m[1, 2] + m[2, 1]) / s_
    else:
        s_ = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s_
        x = (m[0, 2] + m[2, 0]) / s_
        y = (m[1, 2] + m[2, 1]) / s_
        z = 0.25 * s_
    q = torch.stack([w, x, y, z])
    return q / q.norm()


# --------------------------------------------------------------------------

class GNTrackingProbe:
    """Compares Gauss-Newton/LM against Adam on the SAME frame and init."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.frames = list(cfg.get("frames", [50, 200, 400]))
        # HANDOFF SWEEP. Adam runs the non-convex coarse phase, GN finishes.
        # These are the tracking iterations at which a GN probe is launched
        # from whatever pose Adam has reached, so ONE tracking run yields the
        # whole curve of "GN iterations-to-converge vs how long Adam ran".
        self.handoff_iters = list(cfg.get("handoff_iters", [0, 10, 20, 30, 50]))
        self.gn_iters = int(cfg.get("gn_iters", 30))
        self.adam_iters = int(cfg.get("adam_iters", 120))
        self.fd_eps = float(cfg.get("fd_eps", 1e-4))
        self.huber_delta = float(cfg.get("huber_delta", 0.05))
        self.lm_lambda0 = float(cfg.get("lm_lambda0", 1e-3))
        # CONVERGENCE ON ||dxi||, which is P3 of the plan: a step norm is in
        # metres and radians, so it is scene- and scale-meaningful, unlike the
        # fitted loss_eps that has to be retuned per scene. Without this the
        # loop only ever exits by stalling or by budget, and "gn iters" would
        # not measure iterations-to-CONVERGE - which is the whole claim.
        self.step_tol = float(cfg.get("step_tol", 1e-4))
        # TRACKER MODE. The probe measures alongside Adam; this REPLACES it.
        # Everything so far shows GN reaching a lower cost than Adam ever
        # does - but at a pose 0.6-2.9 cm away, and the GT-error delta moves
        # toward truth on one frame and away on another. So "better optimiser"
        # is established and "better tracker" is not. Only a full run answers
        # it, and only ATE decides.
        # OPEN-LOOP EVALUATION. Score the tracker's pose against ground truth,
        # then COMMIT ground truth so the map never inherits tracking error and
        # the next frame starts clean.
        #
        # WHY THIS EXISTS. The closed-loop GN run gave ATE 74.43 cm against
        # Adam's 3.32-3.69, but the dGT trajectory shows a runaway rather than a
        # steady bias: negative (toward truth) for ~25 frames, then positive and
        # growing to +11 cm by frame 43. Tracker and map are coupled, so a
        # tracking failure destroys the very map the objective is defined
        # against, and "GN overfits a good map" cannot be separated from "GN
        # diverged and the map followed it down".
        #
        # Breaking the loop separates them. It is also the standard way to
        # evaluate a tracker in isolation, and it works for ADAM TOO - run it
        # both ways and the comparison is per-frame error on identical maps
        # from identical initialisations, with no accumulation.
        # Motion-prior weights (MAP rather than ML). Zero = the pure ML
        # objective every system currently minimises.
        # prior_rel scales the prior by diag(H) itself, so it is UNIT-FREE and
        # needs no guessing against H's magnitude: each DOF is pulled in
        # proportion to how well the data constrains it. Absolute
        # prior_trans/prior_rot remain for a physically-motivated prior (the
        # inverse covariance of the motion model) once one exists.
        self.prior_rel = float(cfg.get("prior_rel", 0.0))
        self.prior_trans = float(cfg.get("prior_trans", 0.0))
        self.prior_rot = float(cfg.get("prior_rot", 0.0))
        self.open_loop = bool(cfg.get("open_loop", False))
        self.use_as_tracker = bool(cfg.get("use_as_tracker", False))
        self.tracker_handoff = int(cfg.get("tracker_handoff", 0))
        self.tracker_iters = int(cfg.get("tracker_iters", 20))
        self.out_path = cfg.get("out_path", "gn_tracking.jsonl")
        self._records = []
        self._adam = {}      # frame -> Adam's trajectory in the probe's units
        self._final = {}     # frame -> Adam's committed translation
        self._tracked = 0    # frames where GN actually produced the pose
        self._track_log = []  # compact per-frame record in tracker mode
        self._open = []       # open-loop per-frame errors
        # LOSS LINE-SCAN. Independent of the solver - it runs with GN entirely
        # off, on a normal Adam run, and measures the OBJECTIVE rather than any
        # optimiser's path through it.
        self.scan_enabled = bool(cfg.get("line_scan", False))
        self.scan_alphas = [float(a) for a in cfg.get(
            "scan_alphas", [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25])]
        assert 0.0 in self.scan_alphas, "scan_alphas must contain 0.0 (Adam)"
        # every Nth frame; 7-8 extra renders/frame is too much for all of them
        self.scan_every = int(cfg.get("scan_every", 5))
        self._scan = []
        # 6x6 Hessian spectrum, logged per accepted GN step. Free (eigh on a
        # 6x6) and it is the only way to ask whether GN's damage is confined to
        # weakly-observed pose directions.
        self.log_spectrum = bool(cfg.get("log_spectrum", False))
        self._spec = []
        # OBSERVABILITY FILTER. Threshold on lambda_i / lambda_max; 0 disables.
        # "truncate" projects the step out of every direction below it,
        # "damp" shrinks by lambda/(lambda+tau*lambda_max) instead.
        # FD-vs-autograd check. jac_check_iters are GN iterations at which to
        # compare; the hypothesis is about degradation NEAR THE STALL, so the
        # default spans start to saturation (~27) rather than probing only the
        # first step, where the two agree trivially.
        self.jac_check = bool(cfg.get("jac_check", False))
        self.jac_check_iters = [int(x) for x in
                                cfg.get("jac_check_iters", [0, 5, 15, 25])]
        self.jac_eps_sweep = [float(x) for x in cfg.get(
            "jac_eps_sweep", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])]
        self.jac_frames = int(cfg.get("jac_every", 25))
        self._jac = []
        self.spec_tau = float(cfg.get("spec_tau", 0.0))
        self.spec_mode = str(cfg.get("spec_mode", "truncate"))
        assert self.spec_mode in ("truncate", "damp"), \
            f"spec_mode must be truncate|damp, got {self.spec_mode!r}"
        print(f"[GNTrack] constructed (enabled={self.enabled}, "
              f"frames={self.frames}, gn_iters={self.gn_iters}, "
              f"adam_iters={self.adam_iters}, fd_eps={self.fd_eps})",
              flush=True)

    def should_probe(self, time_idx: int) -> bool:
        return self.enabled and time_idx in self.frames

    def should_probe_at(self, time_idx: int, iter_idx: int) -> bool:
        return self.should_probe(time_idx) and iter_idx in self.handoff_iters

    # -- residual ----------------------------------------------------------

    def _residual(self, render_fn, xi, base_w2c, fixed_mask=None):
        """Per-pixel residual vector at pose exp(xi) @ base_w2c.

        Returns (r, w) with r the raw residual and w the IRLS Huber weight.
        GAUSS-NEWTON NEEDS A LEAST-SQUARES OBJECTIVE and SplaTAM's tracking
        loss is L1, so the robust norm is not optional dressing - it is what
        makes the problem well-posed for this solver. Huber with IRLS is what
        DSO and DVO do, and sqrt(w)*r is the least-squares problem whose normal
        equations are J^T W J.
        """
        import torch
        w2c = se3_exp(xi) @ base_w2c
        out = render_fn(w2c)
        r = out["residual"].reshape(-1)
        # FREEZE THE MASK ACROSS A FINITE-DIFFERENCE STENCIL.
        #
        # The valid-pixel set depends on the pose (a Gaussian that projects
        # outside, or whose silhouette drops below threshold, leaves the mask).
        # If it is recomputed at each perturbed pose then r(xi+e) and r(xi-e)
        # are different functions evaluated at different points, and their
        # difference is not a derivative of anything. Direct VO linearises at
        # the current estimate and holds the selection fixed for that step;
        # this does the same.
        valid = out["mask"].reshape(-1) if fixed_mask is None else fixed_mask
        r = r[valid]
        a = r.abs()
        # PER-ELEMENT HUBER DELTA. A single scalar is wrong the moment the
        # residual mixes units: depth is metres (typical |r| ~ 0.01-0.1) and
        # colour is intensity in [0,1]. The closure supplies a delta array
        # scaled the SAME way it scaled the residual, so one comparison is
        # meaningful across both blocks. Falls back to the scalar for
        # single-block residuals.
        d = out.get("huber_delta")
        if d is None:
            d = self.huber_delta
        elif hasattr(d, "reshape"):
            d = d.reshape(-1)[valid]
        w = torch.where(a <= d, torch.ones_like(a), d / a.clamp_min(1e-12))
        return r, w

    @staticmethod
    def _cost(r, w):
        return float((w * r * r).sum() * 0.5)

    def eval_cost(self, render_fn, w2c):
        """The probe's cost at an arbitrary pose.

        Adam minimises a loss in ITS units; the probe measures cost in its own
        (sqrt-weighted, Huber-reweighted). Comparing the two directly would be
        meaningless, so Adam's trajectory is re-evaluated HERE, with the same
        function, and the two curves become comparable.

        The mask is recomputed at each pose rather than frozen - correct for
        evaluating a cost (the real tracking loss does the same), unlike the
        finite-difference stencil where freezing is mandatory.
        """
        import torch
        z = torch.zeros(6, dtype=torch.float32, device=w2c.device)
        r, w = self._residual(render_fn, z, w2c)
        return self._cost(r, w)

    # -- loss line-scan ----------------------------------------------------

    def _scan_cost(self, render_fn, w2c):
        """Raw residual + mask at a pose, kept in (C,H,W) so the channel groups
        stay separable. Cost is NOT computed here - the mask has to be
        intersected across the whole scan first."""
        out = render_fn(w2c)
        r = out["residual"]
        m = out["mask"]
        if m.shape != r.shape:
            m = m.expand_as(r)
        d = out.get("huber_delta")
        if d is None:
            import torch
            d = torch.full_like(r, self.huber_delta)
        return r.detach(), m.detach(), d.detach()

    def line_scan(self, time_idx, render_fn, w2c_adam, gt_w2c):
        """Evaluate the tracking objective ALONG the Adam -> ground-truth line.

        WHY THIS EXISTS. The hybrid run showed GN reducing cost by 3% while
        moving the pose AWAY from ground truth on 85% of frames. That was read
        as "the objective's minimiser is off-target", but LM only guarantees it
        went DOWNHILL - not that it went toward the global minimum. The same
        observation is equally consistent with the minimum sitting near GT and
        GN descending a valley that runs elsewhere. Those two diagnoses call
        for opposite fixes (change the objective vs fix the solve), and nothing
        measured so far separates them.

        This does, and it needs NO optimiser at all. Walk the geodesic

            T(a) = exp(a * d) @ T_adam,   d = log(T_gt @ inv(T_adam))

        so a=0 is Adam's committed pose and a=1 is ground truth, and read off
        where the loss is actually lowest.

            argmin near a=1  -> the objective PREFERS ground truth; the
                                objective is fine and GN's direction is the
                                bug (-> observability filtering).
            argmin near a<=0 -> the objective genuinely prefers a wrong pose;
                                no solver fixes that (-> change the residual).

        THE MASK IS INTERSECTED ACROSS THE WHOLE SCAN, not recomputed per a.
        The valid-pixel set moves with the camera, so a per-a mask would let
        the pixel COUNT change the cost and the curve would partly measure
        "how many pixels are in view" rather than "how well the pose fits".
        Freezing at a=0 alone is also wrong - it would score pixels the map
        does not cover at a=1. The intersection is the only set on which every
        point of the curve is measuring the same thing.

        Reported per channel group because they can disagree, and if depth
        prefers GT while colour prefers elsewhere, that disagreement IS the
        next method.
        """
        import torch
        if not self.enabled:
            return None
        d = se3_log(gt_w2c.to(w2c_adam.dtype) @ torch.linalg.inv(w2c_adam))
        alphas = self.scan_alphas
        res, msk, hdel = [], [], None
        for a in alphas:
            T = se3_exp(float(a) * d) @ w2c_adam
            r, m, hd = self._scan_cost(render_fn, T)
            res.append(r); msk.append(m); hdel = hd
        keep = msk[0]
        for m in msk[1:]:
            keep = keep & m
        nkeep = int(keep.sum())
        # A scan with almost nothing left in the intersection is not evidence
        # of anything; record it and let the summary drop the frame.
        C = res[0].shape[0]
        groups = {"total": slice(0, C)}
        if C == 4:                       # depth ch0, colour ch1-3
            groups["depth"] = slice(0, 1)
            groups["color"] = slice(1, 4)

        def cost(r, sl):
            rr, mm, dd = r[sl], keep[sl], hdel[sl]
            a_ = rr.abs()
            w = torch.where(a_ <= dd, torch.ones_like(a_), dd / a_.clamp_min(1e-12))
            n = int(mm.sum())
            if n == 0:
                return float("nan")
            return float((w * rr * rr)[mm].sum() * 0.5) / n   # PER-PIXEL

        rec = {"frame": int(time_idx), "n_keep": nkeep,
               "alphas": [float(a) for a in alphas],
               "dGT_cm": 100.0 * float(d[:3].norm()),
               "dGT_deg": math.degrees(float(d[3:].norm()))}
        for gname, sl in groups.items():
            rec[gname] = [cost(r, sl) for r in res]
        self._scan.append(rec)
        return rec

    def _line_scan_summary(self) -> str:
        s = [r for r in self._scan if r["n_keep"] > 100]
        if not s:
            return "LOSS LINE-SCAN: enabled but no usable frames"
        al = s[0]["alphas"]
        i0 = al.index(0.0)
        out = [
            f"LOSS LINE-SCAN: Adam -> ground truth, {len(s)} frames"
            f" ({len(self._scan) - len(s)} dropped for empty mask intersection)",
            "  Per-pixel cost on the pixel set valid at EVERY alpha, each frame",
            "  normalised to its own alpha=0 (Adam) value, then MEDIANED.",
            "",
            "  alpha    " + "".join(f"{a:>9.2f}" for a in al),
            "           " + "".join(
                f"{'(Adam)' if i == i0 else ('(GT)' if a == 1.0 else ''):>9}"
                for i, a in enumerate(al)),
        ]
        for g in ("total", "depth", "color"):
            if g not in s[0]:
                continue
            # MEDIAN, not mean, and a RELATIVE denominator guard.
            #
            # These are ratios to each frame's own alpha=0 cost. A frame where
            # Adam happens to sit in a near-zero trough makes that denominator
            # tiny and the ratio explodes - a synthetic check produced 1e13 and
            # a single such frame would swamp a mean over 50 real ones. Frames
            # whose alpha=0 cost is negligible against their own scan range
            # carry no shape information anyway, so they are dropped, and the
            # median makes whatever survives insensitive to the next one.
            ok = [r for r in s
                  if r[g][i0] > 1e-6 * max(x for x in r[g] if not math.isnan(x))]
            col = []
            for i in range(len(al)):
                v = sorted(r[g][i] / r[g][i0] for r in ok
                           if not math.isnan(r[g][i]))
                col.append(v[len(v) // 2] if v else float("nan"))
            out.append(f"  {g:<9}" + "".join(f"{c:>9.4f}" for c in col)
                       + (f"   [{len(s)-len(ok)} frame(s) dropped: degenerate"
                          f" alpha=0 cost]" if len(ok) < len(s) else ""))
        out.append("")
        for g in ("total", "depth", "color"):
            if g not in s[0]:
                continue
            better = sum(1 for r in s if r[g][-1] < r[g][i0])
            # argmin over the scanned grid, averaged - where the objective
            # actually wants the camera to be.
            am = [al[min(range(len(al)), key=lambda i: r[g][i])] for r in s]
            out.append(
                f"  {g:<9} L(GT) < L(Adam) on {better:>3}/{len(s)} frames"
                f"   mean argmin alpha {sum(am)/len(am):>+6.3f}")
        out += [
            "",
            "  READ THE ARGMIN. Near +1 means the objective prefers ground",
            "  truth and GN's DIRECTION is the bug - fixable in the solver.",
            "  Near 0 or negative means the objective prefers a wrong pose and",
            "  no solver can repair it - the residual has to change.",
            "  If depth and colour disagree, that split is the result.",
        ]
        return "\n".join(out)

    def _spectrum_summary(self) -> str:
        s = self._spec
        if not s:
            return "HESSIAN SPECTRUM: enabled but no accepted GN steps"
        # Rank 0 = SMALLEST eigenvalue (worst observed) .. 5 = largest.
        # Aggregating by RANK rather than by absolute eigenvalue is what makes
        # frames comparable: the absolute scale of H moves with the number of
        # valid pixels, the ordering does not.
        n = len(s)
        out = [f"HESSIAN SPECTRUM: {n} accepted GN steps",
               "  rank 0 = weakest-observed pose direction, 5 = strongest.",
               "  gain = squared GT error REMOVED by that direction (cm^2-ish,",
               "  mixed units - read the SIGN and the relative magnitude).",
               "",
               f"  {'rank':<6}{'lambda/lam_max':>16}{'mean gain':>13}"
               f"{'helped':>9}{'|step|':>10}"]
        tot_pos = tot_neg = 0.0
        for k in range(6):
            lam_r = [r["eig"][k] / max(r["eig"][5], 1e-30) for r in s]
            gain = [r["gain"][k] for r in s]
            step = [abs(r["step"][k]) for r in s]
            helped = sum(1 for g in gain if g > 0)
            mg = sum(gain) / n
            tot_pos += sum(g for g in gain if g > 0)
            tot_neg += sum(-g for g in gain if g < 0)
            out.append(f"  {k:<6}{sum(lam_r)/n:>16.3e}{mg:>13.3e}"
                       f"{helped*100.0/n:>8.0f}%{sum(step)/n:>10.3e}")
        out += [
            "",
            f"  total error REMOVED by helping directions {tot_pos:.4e}",
            f"  total error ADDED   by hurting directions {tot_neg:.4e}",
            "",
            "  THE TEST. If the hurting gain is concentrated in ranks 0-2 while",
            "  ranks 3-5 help, GN is right about what it can see and wrong only",
            "  where the frame does not constrain it - truncate or damp the low",
            "  eigendirections and the step becomes usable.",
            "  If ranks 3-5 ALSO hurt, the well-observed directions are being",
            "  driven away from truth and no spectral filter saves it; the",
            "  objective itself is misaligned.",
        ]
        return "\n".join(out)

    # -- finite-difference vs autograd -------------------------------------

    def _autograd_grad(self, render_grad_fn, w2c_cur, mask):
        """Exact d(cost)/d(xi) at w2c_cur, by autograd through the renderer.

        THE HUBER SUBTLETY. The solver's cost is the IRLS surrogate
        0.5*sum(w*r^2) with w = min(1, delta/|r|) held fixed, whose gradient is
        J^T W r. Differentiating that surrogate directly would ALSO
        differentiate w and give a different number. The TRUE Huber loss has
        gradient exactly w*r without holding anything fixed, so it is the one
        that makes autograd and the FD gradient the same quantity. Using the
        surrogate here would manufacture a disagreement that is purely an
        artifact of the comparison.
        """
        import torch
        w2c = w2c_cur.detach().clone().requires_grad_(True)
        out = render_grad_fn(w2c)
        r = out["residual"].reshape(-1)[mask]
        d = out.get("huber_delta")
        d = (d.reshape(-1)[mask] if hasattr(d, "reshape")
             else torch.full_like(r, self.huber_delta))
        a = r.abs()
        loss = torch.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d)).sum()
        loss.backward()
        if w2c.grad is None:
            return None
        G = se3_generators(w2c.dtype, w2c.device)
        Wc = w2c.detach()
        # d cost/d eps_k = <d cost/d w2c , G_k @ w2c>
        return torch.stack([(w2c.grad * (G[k] @ Wc)).sum() for k in range(6)])

    def jacobian_check(self, time_idx, it, render_fn, render_grad_fn, w2c_cur):
        """Is the finite-difference gradient the gradient?

        WHY THIS EXISTS. Every closed-loop GN run exits `stalled`, never
        `converged` - at budget 40, LM rejected all eight damped steps on
        220/249 frames, and the mean iteration count saturates at ~27 no matter
        how large the budget. Stalling is not convergence and not overfitting:
        it means the step the model predicts does not reduce the cost when
        actually evaluated. The obvious suspect is the derivative. As the
        residual approaches its minimum the true gradient shrinks toward zero
        while finite-difference noise does not, so past some point the descent
        direction is numerical noise and LM correctly refuses all of it.
        Explanation (c) in the plan, never tested.
        If FD and autograd disagree, NOTHING measured in this campaign is about
        Gauss-Newton - it is about an approximation to it.

        BOTH SIDES USE LEFT PERTURBATIONS ABOUT w2c_cur. The solver's xi is
        measured from base_w2c, so perturbing xi at xi != 0 and perturbing
        about the current pose differ by the SE(3) Jacobian of the exponential
        map. They coincide only at xi = 0. Re-basing the FD stencil on
        w2c_cur makes the two sides the same quantity at every iteration -
        without it this check would report a growing 'error' that is really
        just the coordinate mismatch, which is exactly the artifact it is meant
        to rule out.

        THE MASK IS FROZEN AND SHARED. Both sides see the identical pixel set.
        """
        import torch
        if render_grad_fn is None:
            return None
        dev, dt = w2c_cur.device, w2c_cur.dtype
        mask = render_fn(w2c_cur)["mask"].reshape(-1)
        g_ag = self._autograd_grad(render_grad_fn, w2c_cur, mask)
        if g_ag is None:
            print("[GNTrack] autograd produced no pose gradient", flush=True)
            return None

        rec = {"frame": int(time_idx), "iter": int(it),
               "autograd": [float(x) for x in g_ag],
               "ag_norm": float(g_ag.norm()), "eps": {}}
        for eps in self.jac_eps_sweep:
            J = torch.zeros((int(mask.sum()), 6), dtype=dt, device=dev)
            ok = True
            for k in range(6):
                e = torch.zeros(6, dtype=dt, device=dev)
                e[k] = eps
                rp, _ = self._residual(render_fn, e, w2c_cur, mask)
                rm, _ = self._residual(render_fn, -e, w2c_cur, mask)
                if rp.numel() != J.shape[0] or rm.numel() != J.shape[0]:
                    ok = False
                    break
                J[:, k] = (rp - rm) / (2 * eps)
            if not ok:
                continue
            r0, w0 = self._residual(render_fn, torch.zeros(6, dtype=dt, device=dev),
                                    w2c_cur, mask)
            g_fd = (J * w0.unsqueeze(1)).t() @ r0
            cos = float(torch.nn.functional.cosine_similarity(
                g_fd.unsqueeze(0), g_ag.unsqueeze(0)).squeeze())
            rec["eps"][f"{eps:g}"] = {
                "cos": cos,
                "ratio": float(g_fd.norm() / g_ag.norm().clamp_min(1e-30)),
                "rel_err": float((g_fd - g_ag).norm()
                                 / g_ag.norm().clamp_min(1e-30)),
            }
        self._jac.append(rec)
        return rec

    def _jac_summary(self) -> str:
        j = self._jac
        if not j:
            return "FD-vs-AUTOGRAD: enabled but never fired"
        eps_keys = list(j[0]["eps"].keys())
        by_it = {}
        for r in j:
            by_it.setdefault(r["iter"], []).append(r)
        out = [
            f"FD-vs-AUTOGRAD JACOBIAN CHECK: {len(j)} probes",
            "  cos = cosine(FD gradient, exact gradient). 1.000 is perfect;",
            "  below ~0.9 the 'descent direction' is substantially noise.",
            "  Both sides are left perturbations about the SAME pose, on the",
            "  SAME frozen mask, against the TRUE Huber loss - so a gap is the",
            "  finite difference and nothing else.",
            "",
        ]
        for it in sorted(by_it):
            rows = by_it[it]
            gn = sum(r["ag_norm"] for r in rows) / len(rows)
            out.append(f"  --- at GN iteration {it}  "
                       f"({len(rows)} frames, mean |exact grad| {gn:.3e}) ---")
            out.append(f"     {'fd_eps':>10}{'cos':>10}{'|fd|/|exact|':>15}"
                       f"{'rel err':>11}")
            for ek in eps_keys:
                v = [r["eps"][ek] for r in rows if ek in r["eps"]]
                if not v:
                    continue
                m = lambda k: sum(x[k] for x in v) / len(v)  # noqa: E731
                out.append(f"     {ek:>10}{m('cos'):>10.4f}"
                           f"{m('ratio'):>15.4f}{m('rel_err'):>11.4f}")
        out += [
            "",
            "  READ THE TREND ACROSS ITERATIONS, not any single row. The",
            "  hypothesis is that FD degrades AS GN APPROACHES ITS STALL - the",
            "  exact gradient shrinks toward zero while FD noise does not. If",
            "  cos is high at iteration 0 and collapses by iteration 25, the",
            "  stalling is a derivative artifact and an analytic Jacobian",
            "  changes the answer. If cos stays high throughout, the FD",
            "  Jacobian is sound and GN genuinely plateaus at ~22 cm.",
            "  fd_eps=1e-4 is what every run in this campaign used; the sweep",
            "  shows whether a different choice would have helped.",
        ]
        return "\n".join(out)

    def note_tracked(self, time_idx):
        self._tracked += 1

    def note_open_loop(self, time_idx, err_cm, err_deg):
        self._open.append({"frame": int(time_idx),
                           "err_cm": float(err_cm), "err_deg": float(err_deg)})

    def _open_loop_summary(self) -> str:
        o = self._open
        if not o:
            return "Open-loop tracking eval: enabled but no frames"
        cm = sorted(r["err_cm"] for r in o)
        dg = sorted(r["err_deg"] for r in o)
        n = len(cm)
        med = lambda v: v[n // 2]  # noqa: E731
        return "\n".join([
            f"OPEN-LOOP TRACKING EVAL: {n} frames. Map built from GROUND TRUTH,"
            f" so there is NO feedback and NO drift accumulation.",
            f"  per-frame translation error cm   mean {sum(cm)/n:.3f}   "
            f"median {med(cm):.3f}   p90 {cm[int(0.9*n)]:.3f}   max {cm[-1]:.3f}",
            f"  per-frame rotation error deg     mean {sum(dg)/n:.3f}   "
            f"median {med(dg):.3f}   p90 {dg[int(0.9*n)]:.3f}   max {dg[-1]:.3f}",
            "",
            "  THIS IS THE COMPARISON THAT DECIDES IT. Run once with",
            "  use_as_tracker=True and once False; identical maps, identical",
            "  initialisations, only the optimiser differs. If GN is worse HERE,",
            "  the objective is genuinely not aligned with pose accuracy. If GN",
            "  matches or beats Adam here, the closed-loop ATE 74 cm was a",
            "  feedback runaway and says nothing about the objective.",
            "  Do NOT read ATE from an open-loop run - the committed poses are",
            "  ground truth, so it is meaningless by construction.",
        ])

    def note_adam(self, time_idx, iter_idx, cost, t):
        """One point of Adam's trajectory on a probed frame."""
        self._adam.setdefault(int(time_idx), []).append(
            {"iter": int(iter_idx), "cost": float(cost),
             "t": [float(x) for x in t]})

    def finalize_frame(self, time_idx, final_w2c):
        """Called once Adam has committed a pose for this frame.

        THE DRIFT-FREE METRIC. Distance to the ground-truth pose conflates this
        frame's tracking error with accumulated global drift, and the drift term
        dominates - a probe run measured 8-9 cm per frame against a final
        aligned ATE of 3.69 cm, and it barely moved over 50 Adam iterations
        because Adam cannot undo an offset inherited from earlier frames.
        Adam's committed pose is what actually produces that ATE, so "does GN
        find the same answer faster" is the meaningful question, and both poses
        live in the same frame.
        """
        tf = [float(x) for x in final_w2c[:3, 3]]
        self._final[int(time_idx)] = tf
        for rec in self._records:
            if rec["frame"] != int(time_idx):
                continue
            for pt in rec["gn"]:
                pt["err_vs_adam"] = 100.0 * math.dist(pt["t"], tf)
        for pt in self._adam.get(int(time_idx), []):
            pt["err_vs_adam"] = 100.0 * math.dist(pt["t"], tf)

    # -- the measurement ---------------------------------------------------

    def run(self, time_idx, render_fn, base_w2c, gt_w2c, adam_fn=None,
            handoff=0, record=True, max_iters=None, render_grad_fn=None):
        """
        render_fn(w2c) -> {"residual": tensor, "mask": bool tensor}
        base_w2c       the pose tracking STARTS from (the motion-model guess)
        gt_w2c         ground-truth pose, for the error curve
        adam_fn()      optional: runs the real Adam loop, returns a list of
                       {"iter", "cost", "w2c"} so both arms are compared on the
                       SAME frame and the SAME initialisation
        """
        import torch
        if not self.enabled:
            return None

        dev = base_w2c.device
        exit_reason = "budget"               # ran out of iterations
        xi = torch.zeros(6, dtype=torch.float32, device=dev)
        lam = self.lm_lambda0
        # Mask is recomputed once per OUTER iteration and held fixed for that
        # iteration's whole stencil.
        mask = render_fn(se3_exp(xi) @ base_w2c)["mask"].reshape(-1)
        r, w = self._residual(render_fn, xi, base_w2c, mask)
        cost = self._cost(r, w)
        _T0 = se3_exp(xi) @ base_w2c
        curve = [{"iter": 0, "cost": cost, "lam": lam,
                  "pose_err": self._pose_err(_T0, gt_w2c),
                  "t": _T0[:3, 3].tolist()}]
        _jac_on = (self.jac_check and render_grad_fn is not None
                   and int(time_idx) % max(self.jac_frames, 1) == 0)
        if _jac_on and 0 in self.jac_check_iters:
            self.jacobian_check(time_idx, 0, render_fn, render_grad_fn, _T0)

        for it in range(1, (max_iters or self.gn_iters) + 1):
            # --- finite-difference J (central). 12 renders. ---------------
            J = torch.zeros((r.numel(), 6), dtype=torch.float32, device=dev)
            for k in range(6):
                e = torch.zeros(6, dtype=torch.float32, device=dev)
                e[k] = self.fd_eps
                rp, _ = self._residual(render_fn, xi + e, base_w2c, mask)
                rm, _ = self._residual(render_fn, xi - e, base_w2c, mask)
                if rp.numel() != r.numel() or rm.numel() != r.numel():
                    # The valid-pixel mask changed size under perturbation, so
                    # the two residual vectors are not the same function
                    # evaluated at two points. Differencing them is
                    # meaningless. This is a REAL failure mode of a masked
                    # photometric residual, not a bug - report and stop rather
                    # than silently differencing mismatched vectors.
                    print(f"[GNTrack] frame {time_idx} iter {it}: mask size "
                          f"changed under perturbation ({rm.numel()}/"
                          f"{r.numel()}/{rp.numel()}); FD Jacobian invalid. "
                          f"Reduce fd_eps or freeze the mask.", flush=True)
                    return
                J[:, k] = (rp - rm) / (2 * self.fd_eps)

            # --- normal equations, 6x6, exact -----------------------------
            Jw = J * w.unsqueeze(1)
            H = J.t() @ Jw                      # 6x6
            g = Jw.t() @ r                      # 6

            # --- MOTION PRIOR. MAP instead of ML. -------------------------
            #
            # The dose-response says pose error rises monotonically as this
            # objective is minimised harder, which - if it is not an artifact
            # of the robustifier - means the objective's minimum is not where
            # the camera is, and the answer is not a weaker optimiser but the
            # MISSING REGULARISER.
            #
            # A Gaussian prior on xi, centred on the motion-model prediction
            # this frame started from, is what classical SLAM puts in the
            # information matrix and what early stopping approximates by
            # refusing to travel far. Making it explicit is the constructive
            # version of the finding: the objective is under-regularised, and
            # the field substitutes a stopping heuristic for a prior.
            #
            # Separate translation and rotation weights because xi mixes metres
            # and radians and one scalar would weight them by accident.
            P = torch.diag(torch.tensor(
                [self.prior_trans] * 3 + [self.prior_rot] * 3,
                dtype=H.dtype, device=H.device))
            if self.prior_rel > 0:
                P = P + self.prior_rel * torch.diag(
                    torch.diag(H).clamp_min(1e-12))
            has_prior = (self.prior_trans > 0 or self.prior_rot > 0
                         or self.prior_rel > 0)

            # --- LM: damp, solve, accept-or-reject ------------------------
            accepted = False
            _xi_pre = xi.clone()
            for _ in range(8):
                Hd = H + P + lam * torch.diag(torch.diag(H).clamp_min(1e-12))
                try:
                    delta = torch.linalg.solve(Hd, -(g + P @ xi))
                except Exception:
                    lam *= 10
                    continue

                # --- OBSERVABILITY FILTER ----------------------------------
                #
                # The spectrum run measured where GN's damage lives: ranks 4-5
                # (lambda/lambda_max 0.74 and 1.00) contributed -1.5e-03 and
                # -2.9e-03 against rank 1's -0.103. Essentially all of it is in
                # directions the frame barely constrains, and the line-scan
                # explains why - Adam already sits at the objective's minimum
                # ALONG the ground-truth direction, so the extra cost GN finds
                # is off that line, in exactly the directions with no curvature
                # to pin them down.
                #
                # Implemented as a PROJECTION of the computed step rather than
                # a modified solve, because that is precisely the counterfactual
                # spec_analysis.py evaluates: gain_i is per-eigendirection and
                # the eigenvectors are orthogonal, so zeroing a component of the
                # step removes exactly that direction's contribution. Rebuilding
                # the solve in the eigenbasis instead would NOT match, since
                # neither P nor lam*diag(H) is diagonal in Q.
                #
                # The point is not to beat Adam. GN reaches the objective's
                # optimum in ~15 iterations against Adam's ~116; if filtering
                # buys PARITY, the iteration ratio is the result.
                if self.spec_tau > 0:
                    try:
                        _ev, _Qf = torch.linalg.eigh(H)
                        _thr = self.spec_tau * float(_ev[-1].clamp_min(1e-30))
                        if self.spec_mode == "damp":
                            # Wiener-style shrinkage: smooth, no cliff at the
                            # threshold, and it cannot discard a direction that
                            # is merely near the boundary.
                            _s = _ev.clamp_min(0) / (_ev.clamp_min(0) + _thr)
                            delta = _Qf @ (_s * (_Qf.t() @ delta))
                        else:                      # "truncate"
                            _k = _ev > _thr
                            if bool(_k.any()):
                                _Qk = _Qf[:, _k]
                                delta = _Qk @ (_Qk.t() @ delta)
                    except Exception as _ex:
                        print(f"[GNTrack] spectral filter failed: {_ex}",
                              flush=True)
                r_new, w_new = self._residual(render_fn, xi + delta, base_w2c, mask)
                if r_new.numel() != r.numel():
                    lam *= 10
                    continue
                c_new = self._cost(r_new, w_new)
                if has_prior:
                    # Compare like with like: the quantity being minimised is
                    # data + prior, so the accept test must include both, or LM
                    # takes steps that lower the data term while the prior term
                    # rises and the "improvement" is fictional.
                    xn = xi + delta
                    c_new = c_new + 0.5 * float(xn @ (P @ xn))
                    cost_cmp = cost + 0.5 * float(xi @ (P @ xi))
                else:
                    cost_cmp = cost
                if c_new < cost_cmp:
                    xi = xi + delta
                    r, w = r_new, w_new
                    # store the DATA cost; the prior term is re-added when
                    # comparing, so the recorded curve stays comparable with
                    # Adam's, which has no prior.
                    cost = self._cost(r_new, w_new)
                    lam = max(lam / 10, 1e-9)
                    accepted = True
                    break
                lam *= 10
            step_norm = float(delta.norm()) if accepted else 0.0
            _T = se3_exp(xi) @ base_w2c

            # --- OBSERVABILITY: where in the spectrum does the damage sit? --
            #
            # H is 6x6, so its eigendecomposition is free, and it answers the
            # question the aggregate dGT number cannot: GN's step may be
            # helping in well-observed directions while a couple of weakly
            # constrained ones wreck the pose. If so the fix is to filter the
            # solve by curvature, not to abandon the solver.
            #
            # In the eigenbasis Q, with e = the left perturbation that would
            # take the PRE-step pose to ground truth, the post-step error along
            # q_i is (e_i - d_i). So
            #     gain_i = e_i^2 - (e_i - d_i)^2
            # is the squared GT error REMOVED by direction i - positive helped,
            # negative hurt. Exact in these coordinates, no linearisation of
            # the residual involved.
            if self.log_spectrum and accepted:
                try:
                    _ev, _Q = torch.linalg.eigh(H)          # ascending
                    _Tpre = se3_exp(_xi_pre) @ base_w2c
                    _e = se3_log(gt_w2c.to(_Tpre.dtype)
                                 @ torch.linalg.inv(_Tpre))
                    _d = xi - _xi_pre                       # accepted step
                    _pe, _pd = _Q.t() @ _e, _Q.t() @ _d
                    _gain = (_pe ** 2 - (_pe - _pd) ** 2)
                    self._spec.append({
                        "frame": int(time_idx), "iter": it,
                        "eig": [float(x) for x in _ev],
                        "step": [float(x) for x in _pd],
                        "to_gt": [float(x) for x in _pe],
                        "gain": [float(x) for x in _gain],
                    })
                except Exception as _ex:      # never let a diagnostic kill a run
                    print(f"[GNTrack] spectrum log failed: {_ex}", flush=True)
            curve.append({"iter": it, "cost": cost, "lam": lam,
                          "pose_err": self._pose_err(_T, gt_w2c),
                          "t": _T[:3, 3].tolist(),
                          "accepted": accepted, "step_norm": step_norm})
            # Fires BEFORE the stall break, so the last probe lands at the pose
            # where LM actually gave up - which is the whole point.
            if _jac_on and it in self.jac_check_iters:
                self.jacobian_check(time_idx, it, render_fn, render_grad_fn, _T)
            if not accepted:
                exit_reason = "stalled"      # LM refused every damped step
                # One probe AT the stall regardless of the schedule: this is
                # the pose the hypothesis is actually about, and it is reached
                # at a different iteration on every frame, so a fixed schedule
                # would almost never sample it.
                if _jac_on and it not in self.jac_check_iters:
                    self.jacobian_check(time_idx, it, render_fn,
                                        render_grad_fn, _T)
                break
            if step_norm < self.step_tol:
                exit_reason = "converged"
                break
            # Re-select valid pixels at the new linearisation point.
            mask = render_fn(se3_exp(xi) @ base_w2c)["mask"].reshape(-1)
            r, w = self._residual(render_fn, xi, base_w2c, mask)
            cost = self._cost(r, w)

        rec = {"frame": int(time_idx), "handoff": int(handoff),
               "exit": exit_reason, "gn": curve}
        if adam_fn is not None:
            rec["adam"] = adam_fn()
        if record:
            self._records.append(rec)
            print(self._summary_one(rec), flush=True)
        else:
            # TRACKER MODE. Recording every frame produced 249 per-frame tables
            # and buried the only number that mattered. Keep a compact
            # aggregate instead.
            self._track_log.append(
                {"frame": int(time_idx), "iters": len(curve) - 1,
                 "exit": exit_reason,
                 "dgt": curve[-1]["pose_err"] - curve[0]["pose_err"],
                 "cost0": curve[0]["cost"], "cost1": curve[-1]["cost"]})
        return xi

    @staticmethod
    def _pose_err(w2c, gt_w2c):
        """Translation error in cm, for a curve that means something."""
        import torch
        if gt_w2c is None:
            return float("nan")
        return float((w2c[:3, 3] - gt_w2c[:3, 3]).norm() * 100.0)

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _summary_one(rec):
        gn = rec["gn"]
        lines = [f"[GNTrack] frame {rec['frame']}",
                 f"    GN/LM : {len(gn)-1} iterations, "
                 f"cost {gn[0]['cost']:.5g} -> {gn[-1]['cost']:.5g}, "
                 f"pose err {gn[0]['pose_err']:.3f} -> {gn[-1]['pose_err']:.3f} cm"]
        if "adam" in rec and rec["adam"]:
            ad = rec["adam"]
            lines.append(f"    Adam  : {len(ad)-1} iterations, "
                         f"cost {ad[0]['cost']:.5g} -> {ad[-1]['cost']:.5g}, "
                         f"pose err {ad[0]['pose_err']:.3f} -> "
                         f"{ad[-1]['pose_err']:.3f} cm")
            # ITERATIONS TO MATCH, which is the actual claim (P1). Comparing
            # final costs after different iteration budgets says nothing.
            target = ad[-1]["cost"]
            hit = next((p["iter"] for p in gn if p["cost"] <= target), None)
            lines.append(f"    -> GN reached Adam's FINAL cost in "
                         f"{hit if hit is not None else '>' + str(len(gn)-1)} "
                         f"iterations (Adam took {len(ad)-1})")
        return "\n".join(lines)

    def save(self):
        # The scan and spectrum streams are written even when _records is
        # empty. A line-scan run produces NOTHING else - gating the whole
        # writer on _records would have silently discarded the entire result.
        # Each stream is tagged so one reader can demultiplex the file.
        if not self.enabled:
            return
        streams = [("gn", self._records), ("scan", self._scan),
                   ("spec", self._spec), ("jac", self._jac)]
        n = sum(len(v) for _, v in streams)
        if n == 0:
            return
        try:
            with open(self.out_path, "w", encoding="utf-8") as fh:
                for kind, recs in streams:
                    for r in recs:
                        fh.write(json.dumps(dict(r, kind=kind)) + "\n")
            print(f"[GNTrack] wrote {n} records "
                  f"({', '.join(f'{k}={len(v)}' for k, v in streams if v)})"
                  f" to {self.out_path}", flush=True)
        except OSError as exc:
            print(f"[GNTrack] could not write {self.out_path}: {exc}", flush=True)

    def _tracker_summary(self) -> str:
        """One block for a whole tracker run.

        Recording per frame produced 249 tables and buried the ATE, which was
        the only number the run existed to produce.
        """
        tl = self._track_log
        n = len(tl) or 1
        conv = sum(1 for r in tl if r["exit"] == "converged")
        stall = sum(1 for r in tl if r["exit"] == "stalled")
        bud = sum(1 for r in tl if r["exit"] == "budget")
        dgt = [r["dgt"] for r in tl]
        worse = sum(1 for d in dgt if d > 0)
        red = sum((r["cost1"] - r["cost0"]) / r["cost0"] for r in tl) / n * 100
        return "\n".join([
            f"GN TRACKER MODE: GN produced the pose for {self._tracked} frames. "
            f"The ATE of this run is GN's, not Adam's.",
            f"  exits      converged {conv}   stalled {stall}   budget {bud}",
            f"  mean iters {sum(r['iters'] for r in tl) / n:.1f}",
            f"  dGT cm     mean {sum(dgt) / n:+.3f}   min {min(dgt):+.3f}   "
            f"max {max(dgt):+.3f}   worse on {worse}/{n} frames",
            f"  cost       mean reduction {red:+.1f}%",
            "",
            "  A negative cost reduction with a positive mean dGT is the",
            "  signature that matters: GN minimised the objective and moved",
            "  the camera AWAY from truth.",
            "  Many `budget` exits mean GN never converged at all, which is a",
            "  different failure and must not be read as the same thing.",
        ])

    def summary(self) -> str:
        if not self.enabled:
            return "GN tracking probe: disabled"
        # DIAGNOSTIC BLOCKS FIRST, and unconditionally. Both are designed to
        # run on configurations where GN produces no pose at all (the line-scan
        # needs no optimiser whatsoever), so gating them behind any of the
        # tracker/open-loop branches below would silently discard the entire
        # result of the run they exist for.
        pre = []
        if self._jac:
            pre.append(self._jac_summary())
        if self._scan:
            pre.append(self._line_scan_summary())
        if self._spec:
            pre.append(self._spectrum_summary())
        if pre:
            rest = self._summary_body()
            return "\n\n".join(pre + ([rest] if rest else []))
        return self._summary_body()

    def _summary_body(self) -> str:
        # TRACKER MODE FIRST. It deliberately keeps no per-frame records, so
        # the "never fired" guard below would swallow the only report the run
        # produces.
        if self._open:
            # The GN tracker's own stats belong here too. Returning only the
            # open-loop block hid them for the entire prior sweep, and they are
            # not cosmetic: with step_tol=0 GN can only end by stalling or by
            # budget, so a strong prior that made it STALL IMMEDIATELY would
            # look like "the prior helps" when it really means "the prior stops
            # GN doing anything". The floor comparison says that is not what
            # happened, but the run should report it rather than leave it to be
            # inferred.
            out = self._open_loop_summary()
            if self._tracked:
                out += "\n\n" + self._tracker_summary()
            return out
        if self._tracked:
            return self._tracker_summary()
        if not self._records:
            return "GN tracking probe: enabled but never fired"
        by_frame = {}
        for r in self._records:
            by_frame.setdefault(r["frame"], []).append(r)
        out = [f"GN tracking probe: {len(by_frame)} frames, "
               f"{len(self._records)} handoff points"]
        for frame in sorted(by_frame):
            adam = self._adam.get(frame, [])
            a_final = adam[-1]["cost"] if adam else float("nan")
            out.append(f"  frame {frame}   Adam: {len(adam)} iters, "
                       f"cost {adam[0]['cost']:.5g} -> {a_final:.5g}"
                       if adam else f"  frame {frame}   (no Adam trajectory)")
            out.append(f"    {'k':>4}{'gn it':>7}{'k+gn':>7}{'adam@cost':>11}"
                       f"{'verdict':>10}{'vs adam cm':>12}{'dGT cm':>9}{'exit':>11}")
            out.append("    " + "-" * 71)
            for r in sorted(by_frame[frame], key=lambda x: x["handoff"]):
                g = r["gn"]
                k, n = r["handoff"], len(g) - 1
                c_gn = g[-1]["cost"]
                # THE CLAIM, MEASURED: how many Adam iterations were needed to
                # reach the cost GN reached? Compare k+n against that. Anything
                # else - final costs after different budgets, or reduction from
                # the handoff point - does not test it.
                m = next((p["iter"] for p in adam if p["cost"] <= c_gn), None)
                if m is None:
                    verdict, m_s = "GN WINS", ">" + str(len(adam) - 1 if adam else 0)
                else:
                    verdict = "GN wins" if (k + n) < m else "no gain"
                    m_s = str(m)
                e = g[-1].get("err_vs_adam", float("nan"))
                # dGT: change in GROUND-TRUTH error across the GN iterations.
                # The absolute value is drift-dominated, but the CHANGE within
                # one frame is not - the inherited offset is constant. Negative
                # means GN moved toward truth. This is the column that
                # separates "better optimiser" from "better tracker", and
                # dropping it in the previous revision was a mistake.
                dgt = g[-1]["pose_err"] - g[0]["pose_err"]
                out.append(f"    {k:>4}{n:>7}{k + n:>7}{m_s:>11}"
                           f"{verdict:>10}{e:>12.3f}{dgt:>+9.3f}"
                           f"{r.get('exit','?'):>11}")
        out.append("")
        out.append("  dGT is the change in ground-truth error DURING GN, which is")
        out.append("  drift-free within a frame. NEGATIVE = moved toward truth.")
        out.append("  Lower cost with positive dGT means GN is a better optimiser")
        out.append("  of an objective that is not aligned with pose accuracy.")
        out.append("  HOW TO READ IT. `adam@cost` is the Adam iteration that first")
        out.append("  reached the cost GN reached; `k+gn` is the hybrid's total. GN")
        out.append("  wins only if k+gn < adam@cost. `>N` means Adam NEVER reached")
        out.append("  it in N iterations, which is GN finding a better optimum.")
        out.append("  `err vs adam` is distance to Adam's COMMITTED pose - drift-free,")
        out.append("  unlike distance to ground truth, which is dominated by offset")
        out.append("  inherited from earlier frames.")
        out.append("")
        out.append("  KILL CRITERION (fixed in advance): if GN needs more than")
        out.append("  ~30 iterations, or only converges with lam so large that")
        out.append("  the step is essentially the gradient direction, the")
        out.append("  quadratic model does not hold on this residual and the")
        out.append("  analytic version is not worth building. Watch `lam`.")
        return "\n".join(out)
