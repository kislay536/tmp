"""Scale-free convergence evidence for pose-tracking optimizers.

The rule deliberately consumes only a loss scalar and an applied tangent step.
Model-specific pose representations must be converted to a common tangent by
the caller, then divided by that optimizer's normal coordinate scales.  The
decision itself therefore has no metres, radians, raw-loss units, or learning
rate in it.
"""

from collections import deque
from math import gcd

import numpy as np


def _skew_numpy(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _se3_exp_numpy(xi):
    """Small CPU SE(3) exponential using the tracker's [rho | theta] order."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho, theta = xi[:3], xi[3:]
    angle = float(np.linalg.norm(theta))
    angle2 = angle * angle
    if angle < 1e-2:
        a = 1.0 - angle2 / 6.0 + angle2 * angle2 / 120.0
        b = 0.5 - angle2 / 24.0 + angle2 * angle2 / 720.0
        c = 1.0 / 6.0 - angle2 / 120.0 + angle2 * angle2 / 5040.0
    else:
        a = np.sin(angle) / angle
        b = (1.0 - np.cos(angle)) / angle2
        c = (angle - np.sin(angle)) / (angle2 * angle)
    w = _skew_numpy(theta)
    w2 = w @ w
    rotation = np.eye(3) + a * w + b * w2
    tangent = np.eye(3) + b * w + c * w2
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = tangent @ rho
    return result


def _se3_inverse_numpy(transform):
    rotation = transform[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ transform[:3, 3])
    return result


def _so3_log_numpy(rotation):
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        # This branch is outside normal tracking motion, but keeping it finite
        # makes a diverged audit useful instead of poisoning all later evidence.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-12:
            return np.zeros(3)
        if pivot == 0:
            axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif pivot == 1:
            axis[0] = np.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = np.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        else:
            axis[0] = np.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = np.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        return angle * axis / max(float(np.linalg.norm(axis)), 1e-12)
    return (0.5 * angle / np.sin(angle)) * vee


def _se3_log_numpy(transform):
    rotation = transform[:3, :3]
    theta = _so3_log_numpy(rotation)
    angle2 = float(theta @ theta)
    angle = np.sqrt(angle2)
    if angle2 < 1e-4:
        coefficient = 1.0 / 12.0 + angle2 / 720.0 + angle2 * angle2 / 30240.0
    elif np.pi - angle < 1e-5:
        coefficient = 1.0 / angle2
    else:
        coefficient = (1.0 / angle2
                       - (1.0 + np.cos(angle))
                       / (2.0 * angle * np.sin(angle)))
    w = _skew_numpy(theta)
    rho = (np.eye(3) - 0.5 * w + coefficient * (w @ w)) @ transform[:3, 3]
    return np.concatenate([rho, theta])


def _pose_distance_numpy(first, second, scales):
    delta = _se3_log_numpy(second @ _se3_inverse_numpy(first))
    return float(np.linalg.norm(delta / scales) / np.sqrt(delta.size))


class WindowedConvergence:
    """Require both pose drift and loss trend to disappear over a window.

    ``pose_score`` is the norm of the mean normalized tangent update divided by
    its standard error. ``loss_score`` is the absolute least-squares loss slope
    divided by its standard error. Coherent motion, improving loss, and
    worsening loss are therefore all evidence to continue. A stop is proposed
    only when BOTH scores stay below the dimensionless ``z_threshold`` for
    ``patience`` checks.

    Shadow mode changes reporting only. The caller decides whether a proposal
    controls its loop by consulting ``active``.
    """

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        self._cfg = dict(cfg)
        self.enabled = bool(cfg.get("enabled", False))
        self.shadow = bool(cfg.get("shadow", True))
        self.window = int(cfg.get("window", 16))
        self.check_every = int(cfg.get("check_every", 4))
        self.patience = int(cfg.get("patience", 2))
        self.z_threshold = float(cfg.get("z_threshold", 2.0))
        self.decay_ratio = float(cfg.get("decay_ratio", 0.25))
        self.proposal_after_phase = int(cfg.get("proposal_after_phase", 0))
        self.eps = float(cfg.get("eps", 1e-12))

        if self.window < 4:
            raise ValueError("windowed convergence window must be >= 4")
        if self.check_every < 1:
            raise ValueError("windowed convergence check_every must be >= 1")
        if self.patience < 1:
            raise ValueError("windowed convergence patience must be >= 1")
        if self.proposal_after_phase < 0:
            raise ValueError("windowed convergence proposal_after_phase must be >= 0")
        if (self.z_threshold <= 0.0 or self.decay_ratio <= 0.0
                or self.eps <= 0.0):
            raise ValueError(
                "windowed convergence z_threshold, decay_ratio and eps must be > 0"
            )

        if scales is None:
            scales = [1.0]
        self.scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        if self.scales.size < 1 or not np.all(np.isfinite(self.scales)):
            raise ValueError("windowed convergence scales must be finite")
        if np.any(self.scales <= 0.0):
            raise ValueError("windowed convergence scales must be positive")

        self._steps = deque(maxlen=self.window)
        self._losses = deque(maxlen=self.window)
        self._frame_open = False
        self._streak = 0
        self._proposed_at = None
        self._proposal_loss = None
        self._frame_best_loss = None
        self._post_step_sum = None
        self._post_best_loss = None
        self._phase_samples = 0
        self._phase_index = 0
        self._phase_reference_energy = None
        self._phase_label = None

        self.frames = 0
        self.proposals = 0
        self.invalid = 0
        self.checks = 0
        self.eligible_checks = 0
        self.energy_rejects = 0
        self.pose_rejects = 0
        self.loss_rejects = 0
        self.patience_waits = 0
        self.phase_resets = 0
        self._proposal_iters = []
        self._proposal_pose_scores = []
        self._proposal_loss_scores = []
        self._proposal_decay_ratios = []
        self._post_motion = []
        self._post_loss_gain = []

    @property
    def active(self):
        return self.enabled and not self.shadow

    def reset_frame(self):
        if not self.enabled:
            return
        if self._frame_open:
            raise RuntimeError("end_frame() must precede the next reset_frame()")
        self.frames += 1
        self._frame_open = True
        self._steps.clear()
        self._losses.clear()
        self._streak = 0
        self._proposed_at = None
        self._proposal_loss = None
        self._frame_best_loss = None
        self._post_step_sum = np.zeros_like(self.scales)
        self._post_best_loss = None
        self._phase_index = 0
        self._begin_phase("frame")

    def start_phase(self, iteration, label):
        """Discard mixed-regime evidence at an optimizer/objective change.

        This is not an iteration floor. Every caller reports phase changes in
        its own tracking loop (optimizer switch/restart, image scale, or pixel
        mask), and the same rule simply requires fresh within-phase evidence.
        """
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede start_phase()")
        if self._proposed_at is not None:
            return
        self.phase_resets += 1
        self._phase_index += 1
        self._begin_phase(f"{label}@{int(iteration)}")

    def _begin_phase(self, label):
        self._steps.clear()
        self._losses.clear()
        self._streak = 0
        self._phase_samples = 0
        self._phase_reference_energy = None
        self._phase_label = str(label)

    def observe(self, iteration, loss, step):
        """Observe one completed update; return True only on a new proposal."""
        if not self.enabled:
            return False
        if not self._frame_open:
            raise RuntimeError("reset_frame() must be called before observe()")

        step = np.asarray(step, dtype=np.float64).reshape(-1)
        loss = float(loss)
        if step.size != self.scales.size:
            raise ValueError(
                f"expected a {self.scales.size}-D step, got {step.size} values"
            )
        if not np.isfinite(loss) or not np.all(np.isfinite(step)):
            # Broken evidence can never mean convergence. Clear the streak and
            # window so finite samples on both sides are not treated as one
            # continuous objective history.
            self.invalid += 1
            self._begin_phase("after-invalid")
            return False

        normalized_step = step / self.scales
        if self._proposed_at is not None:
            # Shadow mode keeps running. Measure how much evidence existed
            # AFTER the proposed stop; a criterion that proposes early and
            # leaves large coherent travel or loss improvement is rejected
            # without needing it to control a trajectory first.
            self._post_step_sum += normalized_step
            self._post_best_loss = (
                loss if self._post_best_loss is None
                else min(self._post_best_loss, loss)
            )
        self._frame_best_loss = (
            loss if self._frame_best_loss is None
            else min(self._frame_best_loss, loss)
        )
        self._steps.append(normalized_step)
        self._losses.append(loss)
        self._phase_samples += 1
        if self._phase_samples == self.window:
            self._phase_reference_energy = self._step_energy(
                np.stack(self._steps)
            )
        completed = int(iteration) + 1
        if (self._proposed_at is not None
                or self._phase_index < self.proposal_after_phase
                or self._phase_samples < 2 * self.window
                or completed % self.check_every != 0):
            return False

        self.eligible_checks += 1
        steps = np.stack(self._steps)
        current_energy = self._step_energy(steps)
        reference = self._phase_reference_energy
        if reference <= self.eps:
            decay_ratio = 0.0 if current_energy <= self.eps else float("inf")
        else:
            decay_ratio = current_energy / reference
        if decay_ratio > self.decay_ratio:
            self.energy_rejects += 1
            self._streak = 0
            return False

        pose_score = self._pose_drift_score(steps)
        loss_score = self._loss_trend_score(np.asarray(self._losses))
        self.checks += 1
        pose_flat = pose_score <= self.z_threshold
        loss_flat = loss_score <= self.z_threshold
        if not pose_flat:
            self.pose_rejects += 1
        if not loss_flat:
            self.loss_rejects += 1
        flat = pose_flat and loss_flat
        self._streak = self._streak + 1 if flat else 0
        if self._streak < self.patience:
            if flat:
                self.patience_waits += 1
            return False

        self._proposed_at = completed
        # The caller commits the best-loss pose seen by the stopping point,
        # not necessarily the proposal row itself. Regret must therefore be
        # measured against that same best-so-far loss.
        self._proposal_loss = self._frame_best_loss
        self.proposals += 1
        self._proposal_iters.append(completed)
        self._proposal_pose_scores.append(pose_score)
        self._proposal_loss_scores.append(loss_score)
        self._proposal_decay_ratios.append(decay_ratio)
        return True

    def end_frame(self):
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede end_frame()")
        if self._proposed_at is not None:
            self._post_motion.append(
                float(np.linalg.norm(self._post_step_sum) / np.sqrt(self.scales.size))
            )
            if self._post_best_loss is None:
                gain = 0.0
            else:
                gain = max(
                    0.0,
                    (self._proposal_loss - self._post_best_loss)
                    / max(abs(self._proposal_loss), self.eps),
                )
            self._post_loss_gain.append(gain)
        self._frame_open = False

    def _pose_drift_score(self, steps):
        mean = steps.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        residual = steps - mean
        variance_energy = float(np.square(residual).sum() / (len(steps) - 1))
        standard_error = np.sqrt(variance_energy / len(steps))
        numerical = self.eps * max(float(np.linalg.norm(steps) / np.sqrt(len(steps))), 1.0)
        if mean_norm <= numerical:
            return 0.0
        return mean_norm / max(standard_error, numerical)

    def _step_energy(self, steps):
        # RMS per tangent coordinate. Both numerator and phase reference use
        # the same caller-supplied coordinate normalization, so their ratio is
        # dimensionless and invariant to model learning-rate units.
        return float(np.sqrt(np.square(steps).sum() / steps.size))

    def _loss_trend_score(self, losses):
        x = np.arange(len(losses), dtype=np.float64)
        x -= x.mean()
        xx = float(x @ x)
        y_mean = float(losses.mean())
        slope = float(x @ (losses - y_mean) / xx)
        residual = losses - (y_mean + slope * x)
        variance = float(residual @ residual / (len(losses) - 2))
        standard_error = np.sqrt(variance / xx)
        numerical = self.eps * max(float(np.max(np.abs(losses))), 1.0)
        if abs(slope) <= numerical:
            return 0.0
        # Absolute slope is intentional: a statistically significant rise is
        # degradation, not convergence, and must not be rewarded with a stop.
        return abs(slope) / max(standard_error, numerical)

    def summary(self, budget=None):
        mode = "shadow" if self.shadow else "active"
        if not self.enabled:
            return "Windowed convergence: disabled"
        base = (f"Windowed convergence: {mode}, window={self.window}, "
                f"every={self.check_every}, patience={self.patience}, "
                f"z<={self.z_threshold:g}, decay<={self.decay_ratio:g}, "
                f"phase>={self.proposal_after_phase}; "
                f"proposals={self.proposals}/{self.frames}, invalid="
                f"{self.invalid}, phase_resets={self.phase_resets}; "
                f"gates eligible={self.eligible_checks}, "
                f"reject energy/pose/loss={self.energy_rejects}/"
                f"{self.pose_rejects}/{self.loss_rejects}, "
                f"patience_wait={self.patience_waits} "
                f"(pose/loss rejects may overlap)")
        if not self._proposal_iters:
            return base
        q = np.percentile(self._proposal_iters, [10, 50, 90])
        pose_med = float(np.median(self._proposal_pose_scores))
        loss_med = float(np.median(self._proposal_loss_scores))
        decay_med = float(np.median(self._proposal_decay_ratios))
        post_motion_q = np.percentile(self._post_motion, [50, 90, 99, 100])
        post_gain_q = np.percentile(self._post_loss_gain, [50, 90, 99, 100])
        result = (base + f"; proposed iteration p10/med/p90="
                  f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}, "
                  f"score med pose/loss={pose_med:.2f}/{loss_med:.2f}, "
                  f"decay med={decay_med:.3g}, post motion "
                  f"med/p90/p99/max={post_motion_q[0]:.3g}/"
                  f"{post_motion_q[1]:.3g}/{post_motion_q[2]:.3g}/"
                  f"{post_motion_q[3]:.3g} units, post loss gain "
                  f"med/p90/p99/max={100 * post_gain_q[0]:.3g}/"
                  f"{100 * post_gain_q[1]:.3g}/"
                  f"{100 * post_gain_q[2]:.3g}/"
                  f"{100 * post_gain_q[3]:.3g}%")
        if budget is not None and int(budget) > 0:
            budget = int(budget)
            cuts = (budget / 3.0, 2.0 * budget / 3.0)
            groups = [[], [], []]
            for i, proposed in enumerate(self._proposal_iters):
                band = 0 if proposed <= cuts[0] else (1 if proposed <= cuts[1] else 2)
                groups[band].append(i)

            def _group(indices):
                if not indices:
                    return "0[-/-]"
                motion = np.median([self._post_motion[i] for i in indices])
                gain = np.median([self._post_loss_gain[i] for i in indices])
                return f"{len(indices)}[{motion:.3g}/{100 * gain:.3g}%]"

            result += ("; proposal bands n[post-motion/loss-gain]: "
                       f"early={_group(groups[0])} "
                       f"mid={_group(groups[1])} "
                       f"late={_group(groups[2])}")
        return result

    def metrics(self, budget=None):
        """Return machine-readable shadow evidence for automatic selection."""
        result = {
            "frames": int(self.frames),
            "proposals": int(self.proposals),
            "proposal_iters": list(self._proposal_iters),
            "post_motion": list(self._post_motion),
            "post_loss_gain": list(self._post_loss_gain),
        }
        if budget is not None and int(budget) > 0:
            budget = int(budget)
            result["mean_saved_per_frame"] = (
                sum(max(0, budget - int(i)) for i in self._proposal_iters)
                / max(self.frames, 1)
            )
        return result


class IncumbentConvergence:
    """Stop when the best-loss pose, rather than the optimizer, is stable.

    The applied steps are composed into an exact relative SE(3) path.  A strict
    loss improvement updates the incumbent pose, but patience is reset only
    when that incumbent moved by ``pose_change`` normalized units or improved
    the anchor loss by ``loss_change``.  Scale-free preconditioner oscillation
    around an unchanged best pose therefore does not keep tracking alive.

    On a full-budget shadow frame, the rule retains its hypothetical incumbent
    and compares it with the final best-loss incumbent.  That counterfactual
    pose regret is the safety measurement used by the automatic controller.
    """

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.shadow = bool(cfg.get("shadow", True))
        self.patience = int(cfg.get("patience", 12))
        self.check_every = int(cfg.get("check_every", 4))
        self.pose_change = float(cfg.get("pose_change", 0.10))
        self.loss_change = float(cfg.get("loss_change", 0.001))
        self.progress_min = float(cfg.get("progress_min", 0.0))
        # ARRIVED-CONVERGED ADMISSION. Iterations of pose stability that let a
        # frame propose even though it never cleared progress_min. 0 disables
        # it, which is the default, so every existing run is bit-identical.
        #
        # WHY progress_min NEEDS AN ESCAPE HATCH. It rejects frames whose TOTAL
        # loss improvement is tiny, to avoid stopping one that is stuck or
        # diverging. On Replica the tracking pose is already at the optimum
        # when the frame opens - constant-velocity extrapolation on a smooth
        # synthetic trajectory - so the whole 60-iteration budget improves the
        # loss by 0.567% and the gate fires 2105 times, committing 59.8 of 60
        # iterations per frame. Those are the SAFEST frames to stop, and the
        # gate blocks exactly them.
        #
        # IT NEEDS ITS OWN MOTION TRACKER, NOT _last_significant. That one is
        # only updated inside `elif loss < self._best_loss`, so on a flat-loss
        # frame it never moves however far the pose travels - and a diverging
        # frame's loss RISES, so it never updates there either. Gating on it
        # would have admitted precisely the frame this is meant to refuse.
        # _last_motion below is updated every observe from the composed pose,
        # so it measures actual displacement independently of the loss.
        #
        # IT CANNOT DISTURB A HEALTHY REGIME. TUM frames run progress of
        # 37-52%, three orders of magnitude above the 0.1% floor, so the branch
        # is unreachable there. It only becomes live where the rule already
        # never fires. Downstream is untouched: the energy rule must still
        # agree and the closed-loop guard still audits and can veto.
        self.stall_patience = int(cfg.get("stall_patience", 0))
        self.proposal_after_phase = int(cfg.get("proposal_after_phase", 0))
        # OPT-IN RELEASE FOR A START-OF-FRAME ANOMALY VETO.  The pose
        # preconditioner's anomaly flag describes only how unusual |g_0| was
        # relative to earlier frames; it stays fixed for the whole frame.  A
        # frame can therefore satisfy the complete incumbent+energy rule over
        # and over and still be forced to the cap.  Count those independently
        # valid proposals and, after N of them, let the current proposal
        # through.  Runtime drift is deliberately different: it describes the
        # steps being taken NOW and remains an absolute veto.
        #
        # Zero preserves the historical hard-veto behaviour exactly.
        self.anomaly_release_after = int(
            cfg.get("anomaly_release_after", 0)
        )
        self.eps = float(cfg.get("eps", 1e-12))
        if self.patience < 1 or self.check_every < 1:
            raise ValueError("incumbent patience and check_every must be >= 1")
        if self.proposal_after_phase < 0:
            raise ValueError("incumbent proposal_after_phase must be >= 0")
        if self.stall_patience < 0:
            raise ValueError("incumbent stall_patience must be >= 0")
        if self.anomaly_release_after < 0:
            raise ValueError("incumbent anomaly_release_after must be >= 0")
        if 0 < self.stall_patience < self.patience:
            # Inert rather than dangerous - the outer gate already requires
            # `patience` - but it always means the config was misread, so say
            # so instead of silently doing nothing.
            raise ValueError(
                f"incumbent stall_patience ({self.stall_patience}) below "
                f"patience ({self.patience}) can never admit anything: the "
                f"proposal gate already requires {self.patience} iterations "
                f"of pose stability. Use 0 to disable, or a value above it.")
        limits = (self.pose_change, self.loss_change, self.progress_min)
        if any(not np.isfinite(value) or value < 0.0 for value in limits):
            raise ValueError("incumbent thresholds must be finite and >= 0")
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("incumbent eps must be finite and > 0")

        self.step_order = str(cfg.get("step_order", "translation_rotation"))
        if self.step_order not in ("translation_rotation", "rotation_translation"):
            raise ValueError("incumbent step_order must name translation/rotation order")
        self.scales = np.asarray(scales if scales is not None else [1.0] * 6,
                                 dtype=np.float64).reshape(-1)
        if self.scales.size != 6 or not np.all(np.isfinite(self.scales)):
            raise ValueError("incumbent convergence requires six finite scales")
        if np.any(self.scales <= 0.0):
            raise ValueError("incumbent convergence scales must be positive")
        if self.step_order == "rotation_translation":
            self.scales = np.concatenate([self.scales[3:], self.scales[:3]])

        # Compatibility fields used by the three tracking-loop integrations.
        self.window = 1
        self.z_threshold = float("inf")
        self.decay_ratio = float("inf")

        self._frame_open = False
        self._phase_index = 0
        self._phase_label = None
        self._current_pose = None
        self._first_loss = None
        self._best_loss = None
        self._best_pose = None
        self._anchor_loss = None
        self._anchor_pose = None
        self._last_significant = 0
        self._last_motion = 0
        self._motion_anchor = None
        self._proposed_at = None
        self._proposal_loss = None
        self._proposal_pose = None
        self._frame_valid = True

        self.frames = 0
        self.proposals = 0
        self.invalid = 0
        self.checks = 0
        self.progress_rejects = 0
        self.stall_admits = 0
        self.barred_proposals = 0
        self._barred_proposal_indices = []
        self.anomaly_releases = 0
        self._frame_anomaly_proposals = 0
        self._frame_anomaly_released = False
        self.phase_resets = 0
        self._proposal_iters = []
        self._proposal_progress = []
        self._post_motion = []
        self._post_loss_gain = []

    @property
    def active(self):
        return self.enabled and not self.shadow

    def reset_frame(self):
        if not self.enabled:
            return
        if self._frame_open:
            raise RuntimeError("end_frame() must precede the next reset_frame()")
        self.frames += 1
        self._frame_open = True
        self._phase_index = 0
        self._phase_label = "frame"
        self._current_pose = np.eye(4)
        self._first_loss = None
        self._best_loss = None
        self._best_pose = None
        self._anchor_loss = None
        self._anchor_pose = None
        self._last_significant = 0
        self._last_motion = 0
        self._motion_anchor = None
        self._proposed_at = None
        self._proposal_loss = None
        self._proposal_pose = None
        self._frame_valid = True
        self._frame_anomaly_proposals = 0
        self._frame_anomaly_released = False

    def start_phase(self, iteration, label):
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede start_phase()")
        if self._proposed_at is not None:
            return
        self.phase_resets += 1
        self._phase_index += 1
        self._phase_label = f"{label}@{int(iteration)}"
        # Require a complete patience interval in the new optimizer/objective
        # regime, but keep the best pose acquired by the earlier regime.
        self._last_significant = max(self._last_significant, int(iteration))
        # A phase change is a new optimizer regime; make the stall path earn a
        # fresh stability run in it rather than inheriting the old one.
        self._last_motion = max(self._last_motion, int(iteration))
        self._motion_anchor = None

    def restart_phase_evidence(self, iteration):
        """Start convergence evidence from the current pose after a rollback.

        A normal optimizer phase change is continuous, so start_phase() keeps
        the incumbent and composed pose. Adaptive full-step rejection is not:
        the caller restores an earlier pose and applies a replacement step.
        Retaining the synthetic pose chain across that jump would turn the
        rollback itself into apparent residual motion and make P8 wait for the
        wrong reason. The first diagonal row establishes a fresh identity
        origin; no threshold or scene unit is introduced.
        """
        if not self.enabled:
            return
        if self._proposed_at is not None:
            raise RuntimeError(
                "cannot restart incumbent evidence after a proposal")
        self._current_pose = np.eye(4)
        self._first_loss = None
        self._best_loss = None
        self._best_pose = None
        self._anchor_loss = None
        self._anchor_pose = None
        self._last_significant = int(iteration)
        self._last_motion = int(iteration)
        self._motion_anchor = None
        self._frame_valid = True

    def _relative_improvement(self, old, new):
        return max(0.0, (old - new) / max(abs(old), self.eps))

    def observe(self, iteration, loss, step, pose_at_loss=None, post_pose=None):
        """Observe loss at the pre-step pose, then compose the applied step."""
        if not self.enabled:
            return False
        if not self._frame_open:
            raise RuntimeError("reset_frame() must be called before observe()")
        loss = float(loss)
        step = np.asarray(step, dtype=np.float64).reshape(-1)
        completed = int(iteration) + 1
        if (step.size != 6 or not np.isfinite(loss)
                or not np.all(np.isfinite(step))):
            self.invalid += 1
            self._frame_valid = False
            return False
        if self.step_order == "rotation_translation":
            step = np.concatenate([step[3:], step[:3]])

        if pose_at_loss is None:
            pose_at_loss = self._current_pose.copy()
        else:
            pose_at_loss = np.asarray(pose_at_loss, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose_at_loss)):
                self.invalid += 1
                self._frame_valid = False
                return False
        if self._first_loss is None:
            self._first_loss = loss
            self._best_loss = loss
            self._best_pose = pose_at_loss
            self._anchor_loss = loss
            self._anchor_pose = pose_at_loss.copy()
            self._last_significant = completed
        elif loss < self._best_loss:
            self._best_loss = loss
            self._best_pose = pose_at_loss
            loss_change = self._relative_improvement(self._anchor_loss, loss)
            # Loss is scalar arithmetic; SE(3) log is not.  Most acquisition
            # updates clear the loss threshold, so test it first and only form
            # a relative transform when loss alone did not reset patience.
            significant = loss_change >= self.loss_change
            if not significant:
                significant = _pose_distance_numpy(
                    self._anchor_pose, pose_at_loss, self.scales
                ) >= self.pose_change
            if significant:
                self._anchor_loss = loss
                self._anchor_pose = pose_at_loss.copy()
                self._last_significant = completed

        # The next row's loss is evaluated at this post-step pose.
        if post_pose is None:
            self._current_pose = _se3_exp_numpy(step) @ pose_at_loss
        else:
            post_pose = np.asarray(post_pose, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(post_pose)):
                self.invalid += 1
                self._frame_valid = False
                return False
            self._current_pose = post_pose.copy()

        # ACTUAL POSE MOTION, every iteration, regardless of the loss. Anchored
        # and re-anchored the same way the incumbent's significance test is, so
        # "moved" means the same thing (pose_change, in the same scaled units)
        # in both places - but this one does not require a better loss to run.
        if self._motion_anchor is None:
            self._motion_anchor = self._current_pose.copy()
        elif _pose_distance_numpy(self._motion_anchor, self._current_pose,
                                  self.scales) >= self.pose_change:
            self._motion_anchor = self._current_pose.copy()
            self._last_motion = completed

        if (not self._frame_valid or self._proposed_at is not None
                or self._phase_index < self.proposal_after_phase
                or completed % self.check_every != 0
                or completed - self._last_significant < self.patience):
            return False
        self.checks += 1
        progress = self._relative_improvement(self._first_loss, self._best_loss)
        if progress < self.progress_min:
            # The frame never improved enough. Admit it anyway ONLY on a long
            # run of real pose stability: `completed - _last_motion` is
            # iterations since the pose last displaced by more than
            # pose_change, measured every iteration and independently of the
            # loss, so a frame that is still moving cannot reach the threshold.
            if (self.stall_patience <= 0
                    or completed - self._last_motion < self.stall_patience):
                self.progress_rejects += 1
                return False
            self.stall_admits += 1

        self._proposed_at = completed
        self._proposal_loss = self._best_loss
        self._proposal_pose = self._best_pose.copy()
        self.proposals += 1
        self._proposal_iters.append(completed)
        self._proposal_progress.append(progress)
        return True

    def end_frame(self):
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede end_frame()")
        if self._proposed_at is not None:
            self._post_motion.append(_pose_distance_numpy(
                self._proposal_pose, self._best_pose, self.scales
            ))
            self._post_loss_gain.append(self._relative_improvement(
                self._proposal_loss, self._best_loss
            ))
        self._frame_open = False

    def note_barred_proposal(self):
        """Record a health-guard veto without changing the local evidence."""
        if self._proposed_at is None or not self._proposal_iters:
            return
        proposal_index = len(self._proposal_iters) - 1
        if (self._barred_proposal_indices
                and self._barred_proposal_indices[-1] == proposal_index):
            return
        self._barred_proposal_indices.append(proposal_index)
        self.barred_proposals += 1

    def health_veto(self, anomalous, drifted):
        """Return whether preconditioner health must veto this proposal.

        ``drifted`` is always fatal for the current proposal.  ``anomalous``
        is a frozen frame-start classification and may be released only after
        the configured number of otherwise-valid proposals in this frame.
        The caller invokes this method only after the full convergence rule
        has proposed, so each count is independent stopping evidence rather
        than a raw iteration.
        """
        if bool(drifted):
            return True
        if not bool(anomalous):
            return False
        if self.anomaly_release_after <= 0:
            return True
        if self._frame_anomaly_released:
            return False

        self._frame_anomaly_proposals += 1
        if self._frame_anomaly_proposals < self.anomaly_release_after:
            return True
        self._frame_anomaly_released = True
        self.anomaly_releases += 1
        return False

    def metrics(self, budget=None):
        result = {
            "frames": int(self.frames),
            "proposals": int(self.proposals),
            "proposal_iters": list(self._proposal_iters),
            "post_motion": list(self._post_motion),
            "post_loss_gain": list(self._post_loss_gain),
            "anomaly_release_after": int(self.anomaly_release_after),
            "anomaly_releases": int(self.anomaly_releases),
        }
        if budget is not None and int(budget) > 0:
            budget = int(budget)
            barred = set(self._barred_proposal_indices)
            result["mean_saved_per_frame"] = (
                sum(
                    max(0, budget - int(value))
                    for index, value in enumerate(self._proposal_iters)
                    if index not in barred
                )
                / max(self.frames, 1)
            )
        return result

    def summary(self, budget=None):
        if not self.enabled:
            return "Incumbent convergence: disabled"
        mode = "shadow" if self.shadow else "active"
        base = (
            f"Incumbent convergence: {mode}, patience={self.patience}, "
            f"every={self.check_every}, pose_change>={self.pose_change:g}, "
            f"loss_change>={100 * self.loss_change:g}%, "
            f"progress>={100 * self.progress_min:g}%, "
            + (f"stall>={self.stall_patience}, " if self.stall_patience > 0 else "")
            + f"phase>={self.proposal_after_phase}; proposals="
            f"{self.proposals}/{self.frames}, invalid={self.invalid}, "
            f"phase_resets={self.phase_resets}, progress_rejects="
            f"{self.progress_rejects}, barred={self.barred_proposals}"
            # Separates "the rule fired normally" from "the rule fired only
            # because the frame arrived converged", which is the whole point of
            # the escape hatch and must not hide inside the proposal count.
            + (f", stall_admits={self.stall_admits}"
               if self.stall_patience > 0 else "")
            + (f", anomaly_release_after={self.anomaly_release_after}, "
               f"released={self.anomaly_releases}"
               if self.anomaly_release_after > 0 else "")
        )
        if not self._proposal_iters:
            return base
        proposed = np.percentile(self._proposal_iters, [10, 50, 90])
        motion = np.percentile(self._post_motion, [50, 90, 100])
        gain = np.percentile(self._post_loss_gain, [50, 90, 100])
        progress = float(np.median(self._proposal_progress))
        base += (
            f"; proposed iteration p10/med/p90={proposed[0]:.0f}/"
            f"{proposed[1]:.0f}/{proposed[2]:.0f}, progress med="
            f"{100 * progress:.3g}%, incumbent regret med/p90/max="
            f"{motion[0]:.3g}/{motion[1]:.3g}/{motion[2]:.3g} units, "
            f"loss regret med/p90/max={100 * gain[0]:.3g}/"
            f"{100 * gain[1]:.3g}/{100 * gain[2]:.3g}%"
        )
        if budget is not None and int(budget) > 0:
            base += (f", saved={self.metrics(budget).get('mean_saved_per_frame', 0.0):.1f}"
                     " iters/frame")
        return base


class IncumbentEnergyConvergence(IncumbentConvergence):
    """Require incumbent stability and phase-relative update-energy decay.

    The incumbent remains the stop proposer.  This class only keeps that
    proposal armed until the recent tangent-update energy has stayed below a
    dimensionless fraction of its post-phase reference.  Flat photometric loss
    therefore cannot stop a tracker whose pose is still moving.
    """

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        super().__init__(cfg, scales=scales)
        self.energy_window = int(cfg.get("energy_window", cfg.get("window", 16)))
        self.energy_patience = int(cfg.get("energy_patience", 2))
        self.decay_ratio = float(cfg.get("decay_ratio", 0.10))
        # PHASE-RELATIVE (default): the reference resets at every start_phase()
        # call, capturing the step size at the moment the CURRENT phase began.
        # FRAME-RELATIVE (energy_phase_relative=False): captured once, at
        # reset_frame(), never re-armed at a later start_phase() - matches
        # what an Adam-only control already measures against (it never calls
        # start_phase() at all, so its reference was always frame-relative).
        self.energy_phase_relative = bool(cfg.get("energy_phase_relative", True))
        if self.energy_window < 4:
            raise ValueError("incumbent-energy window must be >= 4")
        if self.energy_patience < 1:
            raise ValueError("incumbent-energy patience must be >= 1")
        if not np.isfinite(self.decay_ratio) or self.decay_ratio <= 0.0:
            raise ValueError("incumbent-energy decay_ratio must be finite and > 0")

        self.window = self.energy_window
        self._energy_steps = deque(maxlen=self.energy_window)
        self._energy_phase_samples = 0
        self._energy_reference = None
        self._energy_streak = 0
        self._energy_ready = False
        self.energy_checks = 0
        self.energy_rejects = 0
        self.energy_patience_waits = 0
        self._proposal_decay_ratios = []
        self._current_decay_ratio = float("inf")

    def _reset_energy_phase(self):
        self._energy_steps.clear()
        self._energy_phase_samples = 0
        self._energy_reference = None
        self._energy_streak = 0
        self._energy_ready = False
        self._current_decay_ratio = float("inf")

    def reset_frame(self):
        super().reset_frame()
        if self.enabled:
            self._reset_energy_phase()

    def start_phase(self, iteration, label):
        phase_before = self._phase_index
        super().start_phase(iteration, label)
        if (self.enabled and self.energy_phase_relative
                and self._phase_index != phase_before):
            self._reset_energy_phase()

    def _normalized_energy_step(self, step):
        step = np.asarray(step, dtype=np.float64).reshape(-1)
        if step.size != 6 or not np.all(np.isfinite(step)):
            return None
        if self.step_order == "rotation_translation":
            step = np.concatenate([step[3:], step[:3]])
        return step / self.scales

    @staticmethod
    def _step_energy(steps):
        return float(np.sqrt(np.square(steps).sum() / steps.size))

    def _observe_energy(self, iteration, step):
        normalized = self._normalized_energy_step(step)
        if normalized is None:
            self._reset_energy_phase()
            return
        self._energy_steps.append(normalized)
        self._energy_phase_samples += 1
        if self._energy_phase_samples == self.energy_window:
            self._energy_reference = self._step_energy(
                np.stack(self._energy_steps)
            )

        completed = int(iteration) + 1
        if (self._phase_index < self.proposal_after_phase
                or self._energy_phase_samples < 2 * self.energy_window
                or completed % self.check_every != 0):
            return
        current = self._step_energy(np.stack(self._energy_steps))
        reference = self._energy_reference
        if reference is None or reference <= self.eps:
            ratio = 0.0 if current <= self.eps else float("inf")
        else:
            ratio = current / reference
        self._current_decay_ratio = ratio
        self.energy_checks += 1
        if ratio > self.decay_ratio:
            self.energy_rejects += 1
            self._energy_streak = 0
            self._energy_ready = False
            return
        self._energy_streak += 1
        self._energy_ready = self._energy_streak >= self.energy_patience
        if not self._energy_ready:
            self.energy_patience_waits += 1

    def observe(self, iteration, loss, step, pose_at_loss=None, post_pose=None):
        if self.enabled:
            self._observe_energy(iteration, step)
        proposed = super().observe(
            iteration, loss, step, pose_at_loss=pose_at_loss, post_pose=post_pose
        )
        if not proposed:
            return False
        if self._energy_ready:
            self._proposal_decay_ratios.append(self._current_decay_ratio)
            return True

        # The local incumbent is stable but update energy is not.  Keep the
        # proposal armed by undoing only the proposal bookkeeping; all loss,
        # pose and patience evidence remains chronological and can propose at
        # the next aligned check once the energy gate opens.
        self._proposed_at = None
        self._proposal_loss = None
        self._proposal_pose = None
        self.proposals -= 1
        self._proposal_iters.pop()
        self._proposal_progress.pop()
        return False

    def note_barred_proposal(self):
        """Let an active health-vetoed conjunction try again later."""
        if not self.active:
            super().note_barred_proposal()
            return
        if self._proposed_at is None or not self._proposal_iters:
            return
        self.barred_proposals += 1
        self._proposed_at = None
        self._proposal_loss = None
        self._proposal_pose = None
        self.proposals -= 1
        self._proposal_iters.pop()
        self._proposal_progress.pop()
        self._proposal_decay_ratios.pop()

    def summary(self, budget=None):
        base = super().summary(budget).replace(
            "Incumbent convergence:", "Incumbent-energy convergence:", 1
        )
        reference = "phase" if self.energy_phase_relative else "frame"
        energy = (
            f"; energy window={self.energy_window}, reference={reference}, "
            f"decay<={self.decay_ratio:g}, "
            f"patience={self.energy_patience}, checks={self.energy_checks}, "
            f"rejects={self.energy_rejects}, waits={self.energy_patience_waits}"
        )
        if self._proposal_decay_ratios:
            energy += (
                f", proposal decay med="
                f"{float(np.median(self._proposal_decay_ratios)):.3g}"
            )
        return base + energy


class AutoWindowedConvergence:
    """Select and audit one convergence rule without model-specific tuning.

    Calibration frames run to the normal tracking budget. A bank of energy
    rules observes their complete tails, so selection is based on the work and
    best-loss improvement that each hypothetical stop would really have left.
    Normal frames use the fastest safe rule. Periodic full-budget audit frames
    keep collecting the same counterfactual evidence; a selected rule is kept
    while safe and can only be replaced after it violates a safety bound.

    The wrapper intentionally presents the same lifecycle as
    :class:`WindowedConvergence`, keeping SplaTAM, MonoGS and Gaussian-SLAM on
    one implementation. ``shadow=True`` turns the whole controller into a
    diagnostic that calibrates but never controls tracking.
    """

    DEFAULT_SPEC = (
        "w4d25:4:4:2:inf:0.25,w4d20:4:4:2:inf:0.20,"
        "w4d15:4:4:2:inf:0.15,w4d10:4:4:2:inf:0.10,"
        "w4d075:4:4:2:inf:0.075,w4d05:4:4:2:inf:0.05,"
        "w8d25:8:4:2:inf:0.25,w8d20:8:4:2:inf:0.20,"
        "w8d15:8:4:2:inf:0.15,w8d10:8:4:2:inf:0.10,"
        "w8d075:8:4:2:inf:0.075,w8d05:8:4:2:inf:0.05,"
        "w16d25:16:8:2:inf:0.25,w16d20:16:8:2:inf:0.20,"
        "w16d15:16:8:2:inf:0.15,w16d10:16:8:2:inf:0.10,"
        "w16d075:16:8:2:inf:0.075,w16d05:16:8:2:inf:0.05"
    )
    RULE_CLASS = WindowedConvergence
    DISPLAY_NAME = "AutoWindowedConvergence"
    SUMMARY_NAME = "Automatic windowed convergence"

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.shadow = bool(cfg.get("shadow", False))
        self.budget = int(cfg.get("budget", 0))
        self.calibration_skip_frames = int(
            cfg.get("auto_calibration_skip_frames", 0)
        )
        self.calibration_frames = int(cfg.get("auto_calibration_frames", 24))
        self.audit_every = int(cfg.get("auto_audit_every", 50))
        self.audit_patience = int(cfg.get("auto_audit_patience", 1))
        self.min_proposals = int(cfg.get("auto_min_proposals", 4))
        self.loss_p90_limit = float(cfg.get("auto_loss_p90", 0.02))
        self.loss_max_limit = float(cfg.get("auto_loss_max", 0.05))
        self.motion_p90_limit = float(cfg.get("auto_motion_p90", 0.25))
        self.motion_max_limit = float(cfg.get("auto_motion_max", 0.75))
        self.min_saved_fraction = float(cfg.get("auto_min_saved_fraction", 0.0))
        self.proposal_after_phase = int(cfg.get("auto_proposal_after_phase", 0))
        self.report_candidates = bool(cfg.get("auto_report_candidates", False))
        self.final_choice_only = bool(cfg.get("auto_final_choice_only", False))
        if self.budget < 1:
            raise ValueError("automatic windowed convergence requires budget >= 1")
        if self.calibration_frames < 1:
            raise ValueError("auto_calibration_frames must be >= 1")
        if self.calibration_skip_frames < 0:
            raise ValueError("auto_calibration_skip_frames must be >= 0")
        if (self.audit_every < 0 or self.audit_patience < 1
                or self.min_proposals < 1):
            raise ValueError(
                "auto_audit_every must be >= 0; audit patience and "
                "min_proposals must be >= 1"
            )
        limits = (self.loss_p90_limit, self.loss_max_limit,
                  self.motion_p90_limit, self.motion_max_limit,
                  self.min_saved_fraction)
        if any(not np.isfinite(v) or v < 0.0 for v in limits):
            raise ValueError("automatic convergence safety limits must be finite and >= 0")
        if self.min_saved_fraction > 1.0:
            raise ValueError("auto_min_saved_fraction must be <= 1")

        self.scales = np.asarray(scales if scales is not None else [1.0],
                                 dtype=np.float64).reshape(-1)
        self._candidate_cfgs = self._parse_spec(
            cfg.get("auto_spec", self.DEFAULT_SPEC)
        )
        checks = [candidate["check_every"] for candidate in self._candidate_cfgs.values()]
        self.check_every = checks[0]
        for check in checks[1:]:
            self.check_every = gcd(self.check_every, check)
        self.window = min(candidate["window"] for candidate in self._candidate_cfgs.values())
        self.patience = min(candidate["patience"] for candidate in self._candidate_cfgs.values())
        self.z_threshold = float("inf")
        self.decay_ratio = max(
            candidate["decay_ratio"] for candidate in self._candidate_cfgs.values()
        )

        self._evidence_rules = {
            label: self.RULE_CLASS(candidate, scales=self.scales)
            for label, candidate in self._candidate_cfgs.items()
        }
        self._active_rule = None
        self._selected_label = None
        self._selected_snapshot = None
        self._best_rejected = None
        self._frame_open = False
        self._evidence_frame = False
        self._evidence_kind = None
        self._skip_frame = False

        self.frames = 0
        self.calibration_seen = 0
        self.audit_seen = 0
        self.selection_changes = 0
        self.no_safe_rule_events = 0
        self.barred_proposals = 0
        self.audit_deferrals = 0
        self._unsafe_audit_streak = 0
        self._candidate_reported = False

    @property
    def active(self):
        return self.enabled and not self.shadow

    def _parse_spec(self, spec):
        candidates = {}
        for raw in str(spec or "").split(","):
            fields = [field.strip() for field in raw.split(":")]
            if len(fields) != 6 or not all(fields):
                raise ValueError(
                    "WCONV_AUTO_SPEC entries must be label:window:every:patience:z:decay"
                )
            label, window, every, patience, z_threshold, decay_ratio = fields
            if label in candidates:
                raise ValueError(f"duplicate automatic convergence label: {label}")
            candidate = {
                "enabled": True,
                "shadow": True,
                "window": int(window),
                "check_every": int(every),
                "patience": int(patience),
                "z_threshold": float(z_threshold),
                "decay_ratio": float(decay_ratio),
                "proposal_after_phase": self.proposal_after_phase,
            }
            # Let the base rule own detailed validation.
            self.RULE_CLASS(candidate, scales=self.scales)
            candidates[label] = candidate
        if not candidates:
            raise ValueError("automatic windowed convergence needs at least one candidate")
        return candidates

    def reset_frame(self):
        if not self.enabled:
            return
        if self._frame_open:
            raise RuntimeError("end_frame() must precede the next reset_frame()")
        self.frames += 1
        self._frame_open = True
        calibration_end = (
            self.calibration_skip_frames + self.calibration_frames
        )
        self._skip_frame = self.frames <= self.calibration_skip_frames
        if self._skip_frame:
            # Run the unstable prefix at full budget without adding it to any
            # candidate's safety distribution.  There is deliberately no
            # active rule yet, so start_phase/observe/end_frame simply no-op.
            self._evidence_frame = False
            self._evidence_kind = "skip"
            return
        scheduled_audit = (
            self._selected_label is not None
            and self.audit_every > 0
            and self.frames > calibration_end
            and (self.frames - calibration_end) % self.audit_every == 0
        )
        self._evidence_frame = (
            self.shadow
            or self.frames <= calibration_end
            or self._selected_label is None
            or scheduled_audit
        )
        if self._evidence_frame:
            self._evidence_kind = (
                "calibration" if self.frames <= calibration_end
                else "audit"
            )
            for rule in self._evidence_rules.values():
                rule.reset_frame()
        else:
            self._evidence_kind = None
            self._active_rule.reset_frame()

    def start_phase(self, iteration, label):
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede start_phase()")
        if self._skip_frame:
            return
        if self._evidence_frame:
            for rule in self._evidence_rules.values():
                rule.start_phase(iteration, label)
        else:
            self._active_rule.start_phase(iteration, label)

    def observe(self, iteration, loss, step):
        if not self.enabled:
            return False
        if not self._frame_open:
            raise RuntimeError("reset_frame() must be called before observe()")
        if self._skip_frame:
            return False
        if self._evidence_frame:
            for rule in self._evidence_rules.values():
                rule.observe(iteration, loss, step)
            return False
        proposed = self._active_rule.observe(iteration, loss, step)
        return bool(proposed and self.active)

    def end_frame(self):
        if not self.enabled:
            return
        if not self._frame_open:
            raise RuntimeError("reset_frame() must precede end_frame()")
        if self._skip_frame:
            self._frame_open = False
            return
        if self._evidence_frame:
            for rule in self._evidence_rules.values():
                rule.end_frame()
            if self._evidence_kind == "calibration":
                self.calibration_seen += 1
            else:
                self.audit_seen += 1
            if self.calibration_seen >= self.calibration_frames:
                self._select_or_tighten()
        else:
            self._active_rule.end_frame()
        self._frame_open = False

    def note_barred_proposal(self):
        """Record a model-independent preconditioner health-guard veto."""
        self.barred_proposals += 1
        if (not self._evidence_frame and self._active_rule is not None
                and hasattr(self._active_rule, "note_barred_proposal")):
            # The active incumbent rule owns proposal re-arming.  Without
            # this delegation an automatic rule remains latched on its first
            # vetoed proposal and can never accumulate the repeated,
            # independent proposals needed to release a stale anomaly flag.
            self._active_rule.note_barred_proposal()

    def health_veto(self, anomalous, drifted):
        """Apply the selected rule's health policy during runtime.

        Calibration and audit frames never propose to the tracking loop, so
        they retain the conservative hard-veto fallback.  Once a candidate is
        active, delegate to it so ``anomaly_release_after`` can release only a
        frozen frame-start anomaly while keeping live drift absolute.
        """
        if (not self._evidence_frame and self._active_rule is not None
                and hasattr(self._active_rule, "health_veto")):
            return bool(self._active_rule.health_veto(anomalous, drifted))
        return bool(anomalous) or bool(drifted)

    def _candidate_stats(self, label):
        metrics = self._evidence_rules[label].metrics(self.budget)
        losses = np.asarray(metrics["post_loss_gain"], dtype=np.float64)
        motions = np.asarray(metrics["post_motion"], dtype=np.float64)
        if losses.size:
            loss_p90 = float(np.percentile(losses, 90))
            loss_max = float(np.max(losses))
            motion_p90 = float(np.percentile(motions, 90))
            motion_max = float(np.max(motions))
        else:
            loss_p90 = loss_max = motion_p90 = motion_max = 0.0
        safe = (
            metrics["proposals"] >= self.min_proposals
            and metrics.get("mean_saved_per_frame", 0.0)
                >= self.min_saved_fraction * self.budget
            and loss_p90 <= self.loss_p90_limit
            and loss_max <= self.loss_max_limit
            and motion_p90 <= self.motion_p90_limit
            and motion_max <= self.motion_max_limit
        )
        # Maximum limits are hard containment bounds.  A selected rule crosses
        # either one only after a genuinely bad full tail and is rejected on
        # that audit immediately.  P90 is a distribution estimate and moved
        # 0.249 -> 0.252 -> 0.249 in the SplaTAM run; it gets audit patience.
        hard_unsafe = (
            loss_max > self.loss_max_limit
            or motion_max > self.motion_max_limit
        )
        return dict(
            metrics,
            safe=safe,
            hard_unsafe=hard_unsafe,
            loss_p90=loss_p90,
            loss_max=loss_max,
            motion_p90=motion_p90,
            motion_max=motion_max,
        )

    def _rejection_reasons(self, value):
        """Return every selection gate failed by one candidate."""
        reasons = []
        if value["proposals"] < self.min_proposals:
            reasons.append("proposals")
        if (value.get("mean_saved_per_frame", 0.0)
                < self.min_saved_fraction * self.budget):
            reasons.append("saving")
        if value["loss_p90"] > self.loss_p90_limit:
            reasons.append("loss_p90")
        if value["loss_max"] > self.loss_max_limit:
            reasons.append("loss_max")
        if value["motion_p90"] > self.motion_p90_limit:
            reasons.append("motion_p90")
        if value["motion_max"] > self.motion_max_limit:
            reasons.append("motion_max")
        return reasons

    def candidate_frontier(self):
        """Return complete candidate evidence, fastest predicted stop first."""
        rows = []
        for label in self._candidate_cfgs:
            value = self._candidate_stats(label)
            saved = float(value.get("mean_saved_per_frame", 0.0))
            rows.append({
                "candidate": label,
                "predicted_iters": max(0.0, self.budget - saved),
                "saved": saved,
                "proposals": int(value["proposals"]),
                "frames": int(value["frames"]),
                "loss_p90": float(value["loss_p90"]),
                "loss_max": float(value["loss_max"]),
                "motion_p90": float(value["motion_p90"]),
                "motion_max": float(value["motion_max"]),
                "safe": bool(value["safe"]),
                "reasons": self._rejection_reasons(value),
            })
        return sorted(rows, key=lambda row: (row["predicted_iters"], row["candidate"]))

    def _print_candidate_frontier(self):
        prefix = f"[{self.DISPLAY_NAME}Frontier]"
        print(
            f"{prefix}\tBEGIN\tbudget={self.budget}\t"
            f"candidates={len(self._candidate_cfgs)}",
            flush=True,
        )
        print(
            f"{prefix}\tcandidate\tpredicted_iters\tsaved\tproposals\tframes\t"
            "loss_p90_pct\tloss_max_pct\tmotion_p90\tmotion_max\tsafe\treject",
            flush=True,
        )
        for row in self.candidate_frontier():
            reasons = ",".join(row["reasons"]) if row["reasons"] else "-"
            print(
                f"{prefix}\t{row['candidate']}\t{row['predicted_iters']:.2f}\t"
                f"{row['saved']:.2f}\t{row['proposals']}\t{row['frames']}\t"
                f"{100 * row['loss_p90']:.4g}\t{100 * row['loss_max']:.4g}\t"
                f"{row['motion_p90']:.4g}\t{row['motion_max']:.4g}\t"
                f"{int(row['safe'])}\t{reasons}",
                flush=True,
            )
        print(f"{prefix}\tEND", flush=True)

    def _select_or_tighten(self):
        stats = {
            label: self._candidate_stats(label)
            for label in self._candidate_cfgs
        }
        if (self.report_candidates and not self.final_choice_only
                and not self._candidate_reported
                and self.calibration_seen >= self.calibration_frames):
            self._print_candidate_frontier()
            self._candidate_reported = True
        if self._selected_label is not None:
            selected = stats[self._selected_label]
            if selected["safe"]:
                self._selected_snapshot = selected
                self._unsafe_audit_streak = 0
                return
            if (self._evidence_kind == "audit"
                    and not selected["hard_unsafe"]):
                self._unsafe_audit_streak += 1
                if self._unsafe_audit_streak < self.audit_patience:
                    self._selected_snapshot = selected
                    self.audit_deferrals += 1
                    if not self.final_choice_only:
                        print(
                            f"[{self.DISPLAY_NAME}] marginally unsafe audit "
                            f"{self._unsafe_audit_streak}/{self.audit_patience}; "
                            f"keeping {self._selected_label}: loss p90/max="
                            f"{100 * selected['loss_p90']:.3g}/"
                            f"{100 * selected['loss_max']:.3g}%, motion p90/max="
                            f"{selected['motion_p90']:.3g}/"
                            f"{selected['motion_max']:.3g}",
                            flush=True,
                        )
                    return
        prior_saving = None
        if self._selected_label is not None:
            prior_saving = stats[self._selected_label].get(
                "mean_saved_per_frame", 0.0
            )
        safe = [
            (label, value)
            for label, value in stats.items()
            if (value["safe"]
                and (prior_saving is None
                     or value.get("mean_saved_per_frame", 0.0)
                     <= prior_saving + 1e-12))
        ]
        if not safe:
            def _ratio(value, limit):
                if limit > 0.0:
                    return value / limit
                return 0.0 if value <= 0.0 else float("inf")

            rejected_label, rejected = min(
                stats.items(),
                key=lambda item: (
                    max(
                        self.min_proposals / max(item[1]["proposals"], 1),
                        ((self.min_saved_fraction * self.budget)
                         / max(item[1].get("mean_saved_per_frame", 0.0), 1e-12)
                         if self.min_saved_fraction > 0.0 else 0.0),
                        _ratio(item[1]["loss_p90"], self.loss_p90_limit),
                        _ratio(item[1]["loss_max"], self.loss_max_limit),
                        _ratio(item[1]["motion_p90"], self.motion_p90_limit),
                        _ratio(item[1]["motion_max"], self.motion_max_limit),
                    ),
                    -item[1].get("mean_saved_per_frame", 0.0),
                ),
            )
            self._best_rejected = (rejected_label, rejected)
            if self._selected_label is not None and not self.final_choice_only:
                print(
                    f"[{self.DISPLAY_NAME}] audit rejected the selected rule; "
                    "returning to full-budget evidence frames",
                    flush=True,
                )
            self._selected_label = None
            self._selected_snapshot = None
            self._active_rule = None
            self._unsafe_audit_streak = 0
            self.no_safe_rule_events += 1
            if (not self.final_choice_only
                    and (self.no_safe_rule_events == 1
                         or self.no_safe_rule_events
                         % max(self.audit_every, 10) == 0)):
                print(
                    f"[{self.DISPLAY_NAME}] no safe rule after "
                    f"{rejected['frames']} evidence frames; closest="
                    f"{rejected_label}, proposals={rejected['proposals']}/"
                    f"{rejected['frames']}, saved="
                    f"{rejected.get('mean_saved_per_frame', 0.0):.1f}, "
                    f"loss p90/max={100 * rejected['loss_p90']:.3g}/"
                    f"{100 * rejected['loss_max']:.3g}%, motion p90/max="
                    f"{rejected['motion_p90']:.3g}/{rejected['motion_max']:.3g}",
                    flush=True,
                )
            return
        # The objective is saved work per ALL evidence frames, so a rare early
        # proposal cannot beat a broadly useful later one by accident.
        label, chosen = max(
            safe,
            key=lambda item: (
                item[1].get("mean_saved_per_frame", 0.0),
                -item[1]["loss_max"],
                -item[1]["motion_max"],
            ),
        )
        previous = self._selected_label
        self._selected_label = label
        self._selected_snapshot = chosen
        self._best_rejected = None
        active_cfg = dict(self._candidate_cfgs[label], shadow=False)
        self._active_rule = self.RULE_CLASS(active_cfg, scales=self.scales)
        self._unsafe_audit_streak = 0
        self.selection_changes += int(previous != label)
        if not self.final_choice_only:
            print(
                f"[{self.DISPLAY_NAME}] selected {label} after "
                f"{chosen['frames']} evidence frames: proposals="
                f"{chosen['proposals']}/{chosen['frames']}, saved="
                f"{chosen.get('mean_saved_per_frame', 0.0):.1f} iters/frame, "
                f"loss p90/max={100 * chosen['loss_p90']:.3g}/"
                f"{100 * chosen['loss_max']:.3g}%, motion p90/max="
                f"{chosen['motion_p90']:.3g}/{chosen['motion_max']:.3g}",
                flush=True,
            )

    def metrics(self, budget=None):
        """Expose the fixed rule's audit metrics through the auto wrapper.

        Model integrations treat every convergence controller as the same
        interface.  In particular, Gaussian-SLAM's closed-loop guard reads
        the current rule's final-tail regret after a forced audit frame.  A
        normal runtime frame belongs to ``_active_rule``; a calibration or
        automatic-audit frame belongs to the selected shadow rule.  Before a
        rule has been selected there was no controller proposal to audit, so
        return the fixed rule's empty metrics schema.
        """
        metric_budget = self.budget if budget is None else budget
        if (self._evidence_frame
                and self._selected_label in self._evidence_rules):
            return self._evidence_rules[self._selected_label].metrics(
                metric_budget
            )
        if self._active_rule is not None:
            return self._active_rule.metrics(metric_budget)
        return {
            "frames": 0,
            "proposals": 0,
            "proposal_iters": [],
            "post_motion": [],
            "post_loss_gain": [],
            "mean_saved_per_frame": 0.0,
        }

    def summary(self, budget=None):
        if not self.enabled:
            return f"{self.SUMMARY_NAME}: disabled"
        mode = "shadow" if self.shadow else "active"
        selected = self._selected_label or "none"
        base = (
            f"{self.SUMMARY_NAME}: {mode}, calibration="
            f"{self.calibration_frames}, skip={self.calibration_skip_frames}, "
            f"audit_every={self.audit_every}, "
            f"audit_patience={self.audit_patience}, "
            f"candidates={len(self._candidate_cfgs)}, selected={selected}; "
            f"frames={self.frames}, calibration/audit="
            f"{self.calibration_seen}/{self.audit_seen}, changes="
            f"{self.selection_changes}, no_safe={self.no_safe_rule_events}, "
            f"barred={self.barred_proposals}, audit_deferrals="
            f"{self.audit_deferrals}, audit_streak="
            f"{self._unsafe_audit_streak}; "
            f"safety loss p90/max<={100 * self.loss_p90_limit:g}/"
            f"{100 * self.loss_max_limit:g}%, motion p90/max<="
            f"{self.motion_p90_limit:g}/{self.motion_max_limit:g}"
            f", saved>={100 * self.min_saved_fraction:g}% budget"
        )
        if self._selected_snapshot is not None:
            value = self._selected_snapshot
            base += (
                f"; selected evidence proposals={value['proposals']}/"
                f"{value['frames']}, saved={value.get('mean_saved_per_frame', 0.0):.1f} "
                f"iters/frame, loss p90/max={100 * value['loss_p90']:.3g}/"
                f"{100 * value['loss_max']:.3g}%, motion p90/max="
                f"{value['motion_p90']:.3g}/{value['motion_max']:.3g}"
            )
        elif self._best_rejected is not None:
            label, value = self._best_rejected
            base += (
                f"; closest rejected={label}, proposals={value['proposals']}/"
                f"{value['frames']}, saved={value.get('mean_saved_per_frame', 0.0):.1f}, "
                f"loss p90/max={100 * value['loss_p90']:.3g}/"
                f"{100 * value['loss_max']:.3g}%, motion p90/max="
                f"{value['motion_p90']:.3g}/{value['motion_max']:.3g}"
            )
        if self._active_rule is not None:
            runtime = self._active_rule.summary(
                budget if budget is not None else self.budget
            )
            base += "; runtime " + runtime.split(": ", 1)[-1]
        return base


class AutoIncumbentConvergence(AutoWindowedConvergence):
    """Calibrate incumbent stability directly against full-tail pose regret."""

    # 27 candidates: patience controls how long an incumbent must remain
    # stable, pose_change controls meaningful pose motion, and loss_change now
    # spans 0.1/0.5/1%.  The first run fixed it at 0.1%, repeatedly resetting
    # patience on changes whose full-tail loss regret stayed below 1%.
    DEFAULT_SPEC = ",".join(
        f"p{patience}m{int(round(100 * motion)):02d}l"
        f"{int(round(1000 * loss_change)):02d}:"
        f"{patience}:4:{motion:g}:{loss_change:g}:0.001"
        for patience in (4, 16, 32)
        for motion in (0.10, 0.20, 0.40)
        for loss_change in (0.001, 0.005, 0.010)
    )
    RULE_CLASS = IncumbentConvergence
    DISPLAY_NAME = "AutoIncumbentConvergence"
    SUMMARY_NAME = "Automatic incumbent convergence"

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        # A safe rule that barely reduces work defeats the purpose of using the
        # preconditioner.  Energy auto keeps its historical zero default; this
        # pose-incumbent family must save at least a quarter of the budget.
        cfg.setdefault("auto_min_saved_fraction", 0.25)
        cfg.setdefault("auto_audit_patience", 2)
        self.step_order = str(cfg.get("auto_step_order", "translation_rotation"))
        self.anomaly_release_after = int(
            cfg.get("anomaly_release_after", 0)
        )
        self._shared_pose = None
        super().__init__(cfg, scales=scales)

    def reset_frame(self):
        super().reset_frame()
        if self.enabled and self._evidence_frame:
            self._shared_pose = np.eye(4)

    def observe(self, iteration, loss, step):
        if not self.enabled or not self._evidence_frame:
            return super().observe(iteration, loss, step)
        step_array = np.asarray(step, dtype=np.float64).reshape(-1)
        if step_array.size != 6 or not np.all(np.isfinite(step_array)):
            for rule in self._evidence_rules.values():
                rule.observe(iteration, loss, step_array)
            return False
        ordered = step_array
        if self.step_order == "rotation_translation":
            ordered = np.concatenate([step_array[3:], step_array[:3]])
        pose_at_loss = self._shared_pose
        post_pose = _se3_exp_numpy(ordered) @ pose_at_loss
        for rule in self._evidence_rules.values():
            rule.observe(
                iteration, loss, step_array,
                pose_at_loss=pose_at_loss, post_pose=post_pose,
            )
        self._shared_pose = post_pose
        return False

    def _parse_spec(self, spec):
        candidates = {}
        for raw in str(spec or "").split(","):
            fields = [field.strip() for field in raw.split(":")]
            if len(fields) != 6 or not all(fields):
                raise ValueError(
                    "incumbent AUTO_SPEC entries must be "
                    "label:patience:every:pose_change:loss_change:progress"
                )
            label, patience, every, pose_change, loss_change, progress = fields
            if label in candidates:
                raise ValueError(f"duplicate automatic convergence label: {label}")
            candidate = {
                "enabled": True,
                "shadow": True,
                "patience": int(patience),
                "check_every": int(every),
                "pose_change": float(pose_change),
                "loss_change": float(loss_change),
                "progress_min": float(progress),
                "proposal_after_phase": self.proposal_after_phase,
                "step_order": self.step_order,
                "anomaly_release_after": self.anomaly_release_after,
                # Compatibility values consumed by the generic auto wrapper.
                "window": 1,
                "z_threshold": float("inf"),
                "decay_ratio": float("inf"),
            }
            self.RULE_CLASS(candidate, scales=self.scales)
            candidates[label] = candidate
        if not candidates:
            raise ValueError("automatic incumbent convergence needs a candidate")
        return candidates


class AutoIncumbentEnergyConvergence(AutoWindowedConvergence):
    """Calibrate an incumbent-energy decay once, then freeze the result.

    Calibration frames always run to the normal tracking budget.  Every
    candidate sees the same complete loss/pose/update tail, so selection is
    based on the work it would save *and* the best-pose regret it would leave
    behind.  Unlike :class:`AutoWindowedConvergence`, this controller never
    schedules periodic audits.  An optional ``auto_late_audit_frame`` permits
    one later full-budget audit window; ``auto_late_audit_frames`` controls
    how many consecutive frames contribute evidence before the frozen rule is
    reconsidered.  After that window it freezes permanently.
    If no candidate passes the safety filters, it freezes the most conservative
    candidate (the smallest decay, ``0.10`` in the default bank) instead of
    silently reverting to full-budget tracking.

    ``auto_spec`` is deliberately decay-only (``label:decay``).  Window,
    cadence, incumbent patience, and energy patience remain the fixed method;
    changing them at the same time would no longer be automatic decay
    selection.
    """

    DEFAULT_SPEC = (
        "d10:0.10,d15:0.15,d20:0.20,d25:0.25,"
        "d30:0.30,d40:0.40,d50:0.50"
    )
    RULE_CLASS = IncumbentEnergyConvergence
    DISPLAY_NAME = "AutoIncumbentEnergyConvergence"
    SUMMARY_NAME = "Automatic incumbent-energy decay"

    def __init__(self, cfg=None, scales=None):
        cfg = dict(cfg or {})
        requested_audit = int(cfg.get("auto_audit_every", 0))
        if requested_audit != 0:
            raise ValueError(
                "automatic incumbent-energy decay is calibration-only; "
                "auto_audit_every must be 0"
            )
        cfg["auto_audit_every"] = 0
        cfg.setdefault("auto_spec", self.DEFAULT_SPEC)
        self.late_audit_frame = int(cfg.get("auto_late_audit_frame", 0))
        if self.late_audit_frame < 0:
            raise ValueError("auto_late_audit_frame must be >= 0")
        self.late_audit_frames = int(cfg.get("auto_late_audit_frames", 1))
        if self.late_audit_frames < 1:
            raise ValueError("auto_late_audit_frames must be >= 1")
        if self.late_audit_frame == 0 and self.late_audit_frames != 1:
            raise ValueError(
                "auto_late_audit_frames requires auto_late_audit_frame > 0"
            )
        self.late_audit_done = False
        self._late_audit_active = False
        self._late_audit_position = 0

        # Candidate fields shared by every decay arm.  Keep this before the
        # parent constructor: it calls our _parse_spec() dynamically.
        self._decay_candidate_base = {
            "enabled": True,
            "shadow": True,
            "patience": int(cfg.get("patience", 12)),
            "check_every": int(cfg.get("check_every", 4)),
            "pose_change": float(cfg.get("pose_change", 0.10)),
            "loss_change": float(cfg.get("loss_change", 0.001)),
            "progress_min": float(cfg.get("progress_min", 0.0)),
            "stall_patience": int(cfg.get("stall_patience", 0)),
            "anomaly_release_after": int(
                cfg.get("anomaly_release_after", 0)
            ),
            "proposal_after_phase": int(
                cfg.get(
                    "auto_proposal_after_phase",
                    cfg.get("proposal_after_phase", 0),
                )
            ),
            "step_order": str(cfg.get("step_order", "translation_rotation")),
            "energy_window": int(
                cfg.get("energy_window", cfg.get("window", 16))
            ),
            "energy_patience": int(cfg.get("energy_patience", 2)),
            "energy_phase_relative": bool(
                cfg.get("energy_phase_relative", True)
            ),
            # Compatibility fields consumed by the generic auto wrapper.
            "window": int(cfg.get("energy_window", cfg.get("window", 16))),
            "z_threshold": float("inf"),
        }
        self.calibration_frozen = False
        super().__init__(cfg, scales=scales)
        calibration_end = (
            self.calibration_skip_frames + self.calibration_frames
        )
        if 0 < self.late_audit_frame <= calibration_end:
            raise ValueError(
                "auto_late_audit_frame must be after the skipped prefix and "
                "calibration window"
            )

    def _parse_spec(self, spec):
        candidates = {}
        for raw in str(spec or "").split(","):
            fields = [field.strip() for field in raw.split(":")]
            if len(fields) != 2 or not all(fields):
                raise ValueError(
                    "incumbent-energy AUTO_SPEC entries must be label:decay"
                )
            label, decay = fields
            if label in candidates:
                raise ValueError(
                    f"duplicate automatic convergence label: {label}"
                )
            candidate = dict(self._decay_candidate_base)
            candidate["decay_ratio"] = float(decay)
            # Let the real rule validate all shared and candidate values.
            self.RULE_CLASS(candidate, scales=self.scales)
            candidates[label] = candidate
        if not candidates:
            raise ValueError(
                "automatic incumbent-energy decay needs at least one candidate"
            )
        return candidates

    def _select_or_tighten(self):
        if self.calibration_frozen and not self._late_audit_active:
            return
        if (self._late_audit_active
                and self._late_audit_position < self.late_audit_frames):
            if not self.final_choice_only:
                print(
                    f"[{self.DISPLAY_NAME}] late audit evidence "
                    f"{self._late_audit_position}/{self.late_audit_frames} "
                    "collected; keeping the frozen rule until the audit "
                    "window is complete",
                    flush=True,
                )
            return
        super()._select_or_tighten()
        if self.calibration_seen >= self.calibration_frames:
            if self._selected_label is None:
                # Calibration is deliberately one-shot.  The user still asked
                # for stopping to remain active when the evidence is too weak
                # or fails a safety bound, so freeze the least permissive decay
                # rather than reverting to full-budget frames.
                label = min(
                    self._candidate_cfgs,
                    key=lambda name: self._candidate_cfgs[name]["decay_ratio"],
                )
                self._selected_label = label
                self._selected_snapshot = self._candidate_stats(label)
                active_cfg = dict(self._candidate_cfgs[label], shadow=False)
                self._active_rule = self.RULE_CLASS(
                    active_cfg, scales=self.scales
                )
                self.selection_changes += 1
                if not self.final_choice_only:
                    print(
                        f"[{self.DISPLAY_NAME}] no candidate passed the "
                        f"safety filters; freezing conservative fallback "
                        f"{label} (decay={active_cfg['decay_ratio']:.3g})",
                        flush=True,
                    )
            self.calibration_frozen = True
            if self._late_audit_active:
                self.late_audit_done = True
                print(
                    f"[{self.DISPLAY_NAME}] one-time late audit complete: "
                    f"{self._selected_label}; no further audits",
                    flush=True,
                )
            else:
                if self.late_audit_frame > 0:
                    late_end = (
                        self.late_audit_frame + self.late_audit_frames - 1
                    )
                    late_span = (
                        f"at frame {self.late_audit_frame}"
                        if self.late_audit_frames == 1
                        else f"over frames {self.late_audit_frame}-{late_end}"
                    )
                    late = f", one-time late audit {late_span}"
                else:
                    late = ""
                if self.final_choice_only:
                    decay = self._candidate_cfgs[
                        self._selected_label
                    ]["decay_ratio"]
                    print(
                        f"[{self.DISPLAY_NAME}] final choice: "
                        f"{self._selected_label} (decay={decay:.3g})",
                        flush=True,
                    )
                else:
                    print(
                        f"[{self.DISPLAY_NAME}] calibration frozen: "
                        f"{self._selected_label}; periodic audits "
                        f"disabled{late}",
                        flush=True,
                    )

    def reset_frame(self):
        next_frame = self.frames + 1
        self._late_audit_active = (
            self.calibration_frozen
            and self.late_audit_frame > 0
            and not self.late_audit_done
            and self.late_audit_frame <= next_frame
            and next_frame < self.late_audit_frame + self.late_audit_frames
        )
        if not self._late_audit_active:
            return super().reset_frame()
        self._late_audit_position = next_frame - self.late_audit_frame + 1
        if not self.enabled:
            return
        if self._frame_open:
            raise RuntimeError("end_frame() must precede the next reset_frame()")
        self.frames += 1
        self._frame_open = True
        self._evidence_frame = True
        self._evidence_kind = "audit"
        for rule in self._evidence_rules.values():
            rule.reset_frame()
        if self._late_audit_position == 1:
            end_frame = self.late_audit_frame + self.late_audit_frames - 1
            span = (
                f"at frame {self.late_audit_frame}"
                if self.late_audit_frames == 1
                else f"over frames {self.late_audit_frame}-{end_frame}"
            )
            if not self.final_choice_only:
                print(
                    f"[{self.DISPLAY_NAME}] running one-time full-budget "
                    f"late audit {span}",
                    flush=True,
                )

    def summary(self, budget=None):
        base = super().summary(budget)
        state = "frozen" if self.calibration_frozen else "calibrating"
        if self.late_audit_frame > 0:
            end_frame = self.late_audit_frame + self.late_audit_frames - 1
            frame_span = (
                f"frame {self.late_audit_frame}"
                if self.late_audit_frames == 1
                else f"frames {self.late_audit_frame}-{end_frame}"
            )
            late = f"{frame_span} ({'done' if self.late_audit_done else 'pending'})"
        else:
            late = "off"
        return (
            base + f"; decay calibration={state}, periodic audits=disabled, "
            f"one-time late audit={late}"
        )


def make_windowed_convergence(cfg=None, scales=None):
    """Construct the fixed rule or the shared automatic controller."""
    cfg = dict(cfg or {})
    kind = str(cfg.get("auto_kind", cfg.get("kind", "energy"))).lower()
    if kind not in ("energy", "incumbent", "incumbent_energy"):
        raise ValueError(
            "convergence kind must be 'energy', 'incumbent', or "
            "'incumbent_energy'"
        )
    anomaly_release_after = int(cfg.get("anomaly_release_after", 0))
    if anomaly_release_after < 0:
        raise ValueError("anomaly_release_after must be >= 0")
    if anomaly_release_after > 0 and kind == "energy":
        raise ValueError(
            "anomaly_release_after requires an incumbent or "
            "incumbent_energy convergence rule"
        )
    if bool(cfg.get("auto_enabled", False)):
        if kind == "incumbent_energy":
            return AutoIncumbentEnergyConvergence(cfg, scales=scales)
        if kind == "incumbent":
            return AutoIncumbentConvergence(cfg, scales=scales)
        return AutoWindowedConvergence(cfg, scales=scales)
    if kind == "incumbent_energy":
        return IncumbentEnergyConvergence(cfg, scales=scales)
    if kind == "incumbent":
        return IncumbentConvergence(cfg, scales=scales)
    return WindowedConvergence(cfg, scales=scales)


class WindowedConvergenceSweep:
    """Evaluate extra shadow-only parameter choices on identical evidence.

    ``spec`` is a comma-separated list of ``label:z:decay`` entries. ``z=inf``
    intentionally disables the pose/loss significance gates for an energy-only
    shadow diagnostic. The primary rule remains owned by the caller; this
    helper only evaluates the additional candidates, so it can never alter
    loop control or the committed trajectory.
    """

    def __init__(self, spec, base_cfg=None, scales=None):
        self.rules = []
        spec = str(spec or "").strip()
        if not spec:
            return
        base_cfg = dict(base_cfg or {})
        if not bool(base_cfg.get("enabled", False)):
            raise ValueError("windowed convergence sweep requires the primary rule")
        if not bool(base_cfg.get("shadow", True)):
            raise ValueError("windowed convergence sweep is shadow-only")

        labels = set()
        for raw in spec.split(","):
            fields = [field.strip() for field in raw.split(":")]
            if len(fields) != 3 or not all(fields):
                raise ValueError(
                    "WCONV_SWEEP entries must be label:z_threshold:decay_ratio"
                )
            label, z_threshold, decay_ratio = fields
            if label in labels:
                raise ValueError(f"duplicate windowed convergence sweep label: {label}")
            labels.add(label)
            cfg = dict(base_cfg)
            cfg.update(
                enabled=True,
                shadow=True,
                z_threshold=float(z_threshold),
                decay_ratio=float(decay_ratio),
            )
            kind = str(cfg.get("kind", "energy")).lower()
            if kind == "incumbent_energy":
                rule = IncumbentEnergyConvergence(cfg, scales=scales)
            else:
                rule = WindowedConvergence(cfg, scales=scales)
            self.rules.append((label, rule))

    @property
    def enabled(self):
        return bool(self.rules)

    def reset_frame(self):
        for _, rule in self.rules:
            rule.reset_frame()

    def start_phase(self, iteration, label):
        for _, rule in self.rules:
            rule.start_phase(iteration, label)

    def observe(self, iteration, loss, step):
        for _, rule in self.rules:
            rule.observe(iteration, loss, step)

    def end_frame(self):
        for _, rule in self.rules:
            rule.end_frame()

    def summary_lines(self, budget=None):
        return [
            f"Windowed convergence sweep candidate={label}: "
            + rule.summary(budget)
            .removeprefix("Windowed convergence: ")
            .removeprefix("Incumbent-energy convergence: ")
            for label, rule in self.rules
        ]
