"""Per-iteration tracking trace: the pose and pose gradient at EVERY iteration.

Feeds profiling/plot_iter_trace.py, which draws two thesis figures:

  1. consecutive iterations inside a frame are similar - cos(g_k, g_{k+1})
     and the size of the pose step between them;
  2. the pose has settled before the iteration cap - distance of iterate k
     to the pose the frame finally commits, and the first iteration after
     which it stays within tolerance.

Turn on with ITER_TRACE=<path.jsonl> (or tracking.iter_trace.out_path). Off,
every hook is a single attribute check.

ITER_TRACE_EVERY=N (or tracking.iter_trace.every) traces only frames with
frame % N == 0. The run itself still tracks every frame - SplaTAM has to, to
reach the late, hard part of the sequence - but untraced frames pay no syncs
and write nothing. This per-frame switch is host Python, so it assumes the
tracking iteration is NOT captured into a CUDA graph (the trace config turns
both graphs off); under a capture the first frame's decision would be frozen.

NOT A TIMED RUN. Every iteration reads the loss, the pose and the gradient
back to the host, i.e. one sync per iteration. The optimiser arithmetic is
untouched, so trajectories and ATE are those of the traced config.

CAPTURE-SAFE. capture() runs inside the tracking iteration, between backward()
and the step, because the Adam path clears .grad inside step(). It only
copy_()s into buffers allocated once, so it is valid inside a CUDA graph and
replays correctly. Everything that syncs happens in note_iteration(), outside.

One JSON line per tracked frame:
  {"frame", "cap", "init": {q, t}, "committed": {q, t},
   "iters": [{"it", "loss", "reused", "grad": [rho|theta] or null,
              "q", "t"}]}      # q, t = pose AFTER that iteration's step
The gradient is dL/dxi in the SE(3) tangent at the pre-step pose, [rho|theta]
order, same map the preconditioner uses (quat_trans_grad_to_tangent).
"""

import json
import os

import torch

from utils.pose_preconditioner import quat_trans_grad_to_tangent


def rot_to_quat(R):
    """(w, x, y, z) of a 3x3 rotation, float64 CPU tensor (Shepperd). For a
    model that stores R rather than a quaternion (MonoGS)."""
    R = R.detach().double().cpu()
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        s = 2.0 * (tr + 1.0) ** 0.5
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * float(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * float(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * float(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = torch.tensor([float(v) for v in q], dtype=torch.float64)
    return q / q.norm()


def _host(x):
    return [float(v) for v in x.detach().reshape(-1).cpu().tolist()]


class IterTrace:
    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        path = os.environ.get("ITER_TRACE", "").strip() or cfg.get("out_path", "")
        self.enabled = bool(path) and path not in ("0", "false", "False")
        self.path = path
        self.every = max(1, int(os.environ.get("ITER_TRACE_EVERY", "")
                                or cfg.get("every", 1)))
        self._gbuf = None
        self._frame = None
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w")
            print(f"[IterTrace] writing per-iteration pose/grad trace to {path}, "
                  f"every {self.every} frame(s) (one host sync per traced "
                  "iteration - not a timed run)", flush=True)

    def wants(self, frame):
        """Whether begin_frame(frame) will trace. Lets a caller skip building
        the pose it would pass (Gaussian-SLAM composes an absolute one)."""
        return self.enabled and int(frame) % self.every == 0

    @property
    def active(self):
        return self.enabled and self._frame is not None

    def begin_frame(self, frame, cap, q, t):
        if not self.enabled:
            return
        if int(frame) % self.every != 0:
            self._frame = None
            return
        if self._gbuf is None:
            # Allocated ONCE. A captured iteration bakes this address in.
            self._gbuf = torch.zeros(6, dtype=torch.float64, device=q.device)
        self._frame = {"frame": int(frame), "cap": int(cap),
                       "init": {"q": _host(q), "t": _host(t)}, "iters": []}

    def capture(self, q, t, g_q, g_t, tangent_map=None):
        """Inside the iteration, after backward(), before the step.

        tangent_map (6x6, optional) is applied to the tangent gradient, for a
        model whose optimised variable is not the absolute w2c. Gaussian-SLAM
        optimises L in W = A L with A fixed per frame; a left perturbation of L
        is xi_abs = Ad_A xi_rel, so g_abs = Ad_{A^-1}^T g_rel.
        """
        if (not self.enabled or self._frame is None
                or g_q is None or g_t is None):
            return
        with torch.no_grad():
            g = quat_trans_grad_to_tangent(q.reshape(4), t.reshape(3),
                                           g_q.reshape(4), g_t.reshape(3))
            g = g.reshape(6).to(self._gbuf.dtype)
            if tangent_map is not None:
                g = tangent_map.to(self._gbuf.dtype) @ g
            self._gbuf.copy_(g)

    def capture_tangent(self, g_rho, g_theta):
        """For a model whose optimised variable IS the left tangent of the
        absolute w2c - MonoGS applies T <- exp([rho; theta]) T and zeroes the
        deltas every step, so their gradients need no mapping. [rho | theta]
        order, i.e. cam_trans_delta.grad then cam_rot_delta.grad."""
        if (not self.enabled or self._frame is None
                or g_rho is None or g_theta is None):
            return
        with torch.no_grad():
            self._gbuf.copy_(torch.cat([g_rho.reshape(3), g_theta.reshape(3)])
                             .to(self._gbuf.dtype))

    def note_iteration(self, it, loss, q, t, reused=False):
        """After the iteration (and its step) has run. Syncs."""
        if not self.enabled or self._frame is None:
            return
        self._frame["iters"].append({
            "it": int(it),
            "loss": float(loss.detach()) if torch.is_tensor(loss) else float(loss),
            "reused": bool(reused),
            # A reuse iteration never ran backward, so the buffer still holds
            # the previous gradient - recording it would fake cos = 1.
            "grad": None if reused else _host(self._gbuf),
            "q": _host(q), "t": _host(t),
        })

    def end_frame(self, q, t):
        if not self.enabled or self._frame is None:
            return
        self._frame["committed"] = {"q": _host(q), "t": _host(t)}
        self._fh.write(json.dumps(self._frame) + "\n")
        self._fh.flush()
        self._frame = None

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
