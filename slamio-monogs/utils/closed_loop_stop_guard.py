"""Closed-loop supervision for an otherwise local tracking stop rule.

The convergence rule answers a within-frame question: would its incumbent be
close to the incumbent obtained at the normal tracking budget?  In SLAM, that
is necessary but not sufficient.  Committing an early pose changes mapping and
keyframe selection, so the next frame is no longer sampled from the shadow
trajectory on which the rule was validated.

``AuditedStopGuard`` keeps the stop rule unchanged and supervises only whether
an active proposal may control the loop:

* initial full-budget frames build scale-free coverage/motion history;
* every Nth eligible proposal is allowed to run to the full budget, retaining
  a real counterfactual tail (N=0 keeps only the initial calibration tails);
* unsafe audit regret latches the run back to full-budget tracking;
* robust coverage and motion-prior-innovation outliers veto a proposal;
* a critically collapsed final render may recover the frame's original motion
  prediction instead of teaching the map a zero-loss, out-of-frustum pose.

The class contains no model or torch dependency, so its state machine can be
unit-tested without a GPU.  The caller owns all pose/coverage measurements.
"""

from collections import deque

import numpy as np


class AuditedStopGuard:
    """Decide whether an active convergence proposal stops, audits, or vetoes."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.calibration_frames = int(cfg.get("calibration_frames", 24))
        self.audit_every = int(cfg.get("audit_every_proposals", 8))
        self.history_size = int(cfg.get("history_size", 64))
        self.min_history = int(cfg.get("min_history", 16))
        self.robust_z = float(cfg.get("robust_z", 6.0))
        self.coverage_ratio = float(cfg.get("coverage_ratio", 0.50))
        self.critical_coverage_ratio = float(
            cfg.get("critical_coverage_ratio", 0.20)
        )
        self.audit_motion_p90 = float(cfg.get("audit_motion_p90", 0.25))
        self.audit_motion_max = float(cfg.get("audit_motion_max", 0.75))
        self.audit_loss_p90 = float(cfg.get("audit_loss_p90", 0.02))
        self.audit_loss_max = float(cfg.get("audit_loss_max", 0.05))
        self.audit_min_samples = int(cfg.get("audit_min_samples", 4))
        # RELEASE, OFF BY DEFAULT (0). The latch is a one-way switch without
        # it - measured on GSLAM/fr1_desk, one collapse near frame ~365 kept
        # 225/590 frames (38.1%) on full-budget audits for the rest of a
        # 590-frame run, an estimated ~37.5s (~6.7% of tracking time) that a
        # healthy tail could have recovered. release_after>0 lets that many
        # CONSECUTIVE clean latched audits release the latch. See end_frame()
        # for why this needs its own, local notion of "clean" rather than
        # reusing _audit_unsafe().
        self.release_after = int(cfg.get("release_after", 0))

        if self.calibration_frames < 0:
            raise ValueError("guard calibration_frames must be >= 0")
        if self.audit_every < 0:
            raise ValueError("guard audit_every_proposals must be >= 0")
        if self.history_size < 1 or not 1 <= self.min_history <= self.history_size:
            raise ValueError("guard history requires 1 <= min_history <= history_size")
        if not np.isfinite(self.robust_z) or self.robust_z <= 0.0:
            raise ValueError("guard robust_z must be finite and > 0")
        ratios = (self.coverage_ratio, self.critical_coverage_ratio)
        if any(not np.isfinite(v) or not 0.0 < v <= 1.0 for v in ratios):
            raise ValueError("guard coverage ratios must be in (0, 1]")
        limits = (
            self.audit_motion_p90,
            self.audit_motion_max,
            self.audit_loss_p90,
            self.audit_loss_max,
        )
        if any(not np.isfinite(v) or v < 0.0 for v in limits):
            raise ValueError("guard audit limits must be finite and >= 0")
        if self.audit_min_samples < 1:
            raise ValueError("guard audit_min_samples must be >= 1")
        if self.release_after < 0:
            raise ValueError("guard release_after must be >= 0")

        self._coverage_history = deque(maxlen=self.history_size)
        self._motion_history = deque(maxlen=self.history_size)
        self._audit_motion = []
        self._audit_loss = []

        self.frames = 0
        self.proposals = 0
        self.eligible_proposals = 0
        self.active_stops = 0
        self.calibration_audits = 0
        self.periodic_audits = 0
        self.latched_audits = 0
        self.coverage_vetoes = 0
        self.motion_vetoes = 0
        self.audit_failures = 0
        self.recoveries = 0
        self.releases = 0
        self.latched_full_budget = False
        self.latch_reason = "-"
        # CONSECUTIVE clean latched audits since the last latch/veto/reset -
        # a run-level counter, not per-frame: it has to survive across
        # reset_frame() calls to mean "N in a row", the same way
        # latched_full_budget does.
        self._clean_streak = 0

        self._frame_action = "full"
        self._frame_audit = False
        self._frame_health_veto = False

    @staticmethod
    def _finite(value):
        return value is not None and np.isfinite(float(value))

    @staticmethod
    def _robust_center_scale(values, relative_floor):
        values = np.asarray(values, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q75 = float(np.percentile(np.abs(values), 75))
        floor = relative_floor * max(abs(median), q75, 1.0)
        scale = max(1.4826 * mad, floor, 1e-12)
        return median, scale

    def reset_frame(self):
        if not self.enabled:
            return
        self.frames += 1
        self._frame_action = "full"
        self._frame_audit = False
        self._frame_health_veto = False

    def _health_reasons(self, coverage, innovation):
        reasons = []
        if (len(self._coverage_history) >= self.min_history
                and self._finite(coverage)):
            median, scale = self._robust_center_scale(
                self._coverage_history, relative_floor=0.05
            )
            lower = max(
                self.coverage_ratio * median,
                median - self.robust_z * scale,
            )
            if float(coverage) < lower:
                reasons.append("coverage")
        if (len(self._motion_history) >= self.min_history
                and self._finite(innovation)):
            median, scale = self._robust_center_scale(
                self._motion_history, relative_floor=0.10
            )
            if float(innovation) > median + self.robust_z * scale:
                reasons.append("motion")
        return reasons

    def decide_proposal(self, coverage, innovation, can_stop=True):
        """Return ``stop``, ``audit``, or ``veto`` for one local proposal."""
        if not self.enabled:
            return "stop"
        self.proposals += 1

        calibrating = (
            self.frames <= self.calibration_frames
            or len(self._coverage_history) < self.min_history
            or len(self._motion_history) < self.min_history
        )

        # CALIBRATION-ONLY MODE. The initial full tails are retained to build
        # the requested history, but no later proposal is held for a periodic
        # counterfactual audit, health veto, or full-budget latch. This is
        # deliberately stronger than merely choosing a very large audit
        # interval: a critical-render recovery must not silently re-arm the
        # full-budget latch after calibration either.
        if self.audit_every == 0 and not calibrating:
            self.eligible_proposals += 1
            self.active_stops += 1
            self._frame_action = "stop"
            return "stop"

        # One unhealthy proposal makes this a full-budget recovery frame. Do
        # not let a later re-armed proposal stop after a few superficially
        # healthier rows and erase the countermeasure.
        if self._frame_health_veto:
            return "veto"

        reasons = self._health_reasons(coverage, innovation)
        if reasons:
            self.coverage_vetoes += int("coverage" in reasons)
            self.motion_vetoes += int("motion" in reasons)
            self._frame_action = "health-veto"
            self._frame_health_veto = True
            vetoes = self.coverage_vetoes + self.motion_vetoes
            if vetoes <= 5 or vetoes % 25 == 0:
                print(
                    f"[ClosedLoopStopGuard] frame {self.frames}: proposal "
                    f"vetoed by {','.join(reasons)} health "
                    f"(coverage={float(coverage):.4g}, "
                    f"innovation={float(innovation):.4g})",
                    flush=True,
                )
            return "veto"

        # A proposal discovered only in the terminal drain has no work left to
        # save.  Keep its full tail as evidence instead of counting it active.
        if not can_stop:
            self._frame_action = "terminal-audit"
            self._frame_audit = True
            return "audit"

        if self.latched_full_budget:
            self.latched_audits += 1
            self._frame_action = "latched-audit"
            self._frame_audit = True
            return "audit"

        if calibrating:
            self.calibration_audits += 1
            self._frame_action = "calibration-audit"
            self._frame_audit = True
            return "audit"

        self.eligible_proposals += 1
        if self.eligible_proposals % self.audit_every == 0:
            self.periodic_audits += 1
            self._frame_action = "periodic-audit"
            self._frame_audit = True
            print(
                f"[ClosedLoopStopGuard] frame {self.frames}: eligible "
                f"proposal {self.eligible_proposals} is periodic full-tail "
                f"audit {self.periodic_audits}",
                flush=True,
            )
            return "audit"

        self.active_stops += 1
        self._frame_action = "stop"
        return "stop"

    def _audit_unsafe(self):
        if not self._audit_motion:
            return False, "-"
        motion = np.asarray(self._audit_motion, dtype=np.float64)
        loss = np.asarray(self._audit_loss, dtype=np.float64)
        motion_max = float(np.max(motion))
        loss_max = float(np.max(loss))
        if motion_max > self.audit_motion_max:
            return True, f"audit-motion-max={motion_max:.3g}"
        if loss_max > self.audit_loss_max:
            return True, f"audit-loss-max={100 * loss_max:.3g}%"
        if motion.size >= self.audit_min_samples:
            motion_p90 = float(np.percentile(motion, 90))
            loss_p90 = float(np.percentile(loss, 90))
            if motion_p90 > self.audit_motion_p90:
                return True, f"audit-motion-p90={motion_p90:.3g}"
            if loss_p90 > self.audit_loss_p90:
                return True, f"audit-loss-p90={100 * loss_p90:.3g}%"
        return False, "-"

    def end_frame(self, *, full_budget, coverage, innovation,
                  audit_motion=None, audit_loss=None):
        """Close a frame, update trusted history, and evaluate any audit."""
        if not self.enabled:
            return
        if full_budget and self._finite(coverage) and self._finite(innovation):
            self._coverage_history.append(float(coverage))
            self._motion_history.append(float(innovation))

        if (self.audit_every > 0 and self._frame_audit
                and self._finite(audit_motion)
                and self._finite(audit_loss)):
            self._audit_motion.append(float(audit_motion))
            self._audit_loss.append(float(audit_loss))
            unsafe, reason = self._audit_unsafe()
            if unsafe and not self.latched_full_budget:
                self.audit_failures += 1
                self.latched_full_budget = True
                self.latch_reason = reason
                print(
                    "[ClosedLoopStopGuard] unsafe counterfactual audit; "
                    f"latching full-budget tracking ({reason})",
                    flush=True,
                )

            # RELEASE. _audit_unsafe() above cannot be reused here: it takes
            # a MAX/percentile over the ENTIRE audit history, so once any one
            # bad sample has occurred it reports unsafe FOREVER by
            # construction - that permanence is exactly what makes it safe to
            # ARM with, but it also means a streak counted against it could
            # never grow past zero. Release instead asks a LOCAL question:
            # does THIS audit's own sample, on its own, clear the same hard
            # ceiling used to arm the latch? release_after consecutive clean
            # samples release it; one sample that fails the ceiling zeroes
            # the streak, and so does a fresh should_recover_initial()
            # collapse (a second catastrophic event during probation voids
            # whatever confidence the streak had built).
            if self.latched_full_budget and self.release_after > 0:
                clean = (float(audit_motion) <= self.audit_motion_max
                         and float(audit_loss) <= self.audit_loss_max)
                if clean:
                    self._clean_streak += 1
                    if self._clean_streak >= self.release_after:
                        self.latched_full_budget = False
                        self.releases += 1
                        self._clean_streak = 0
                        print(
                            f"[ClosedLoopStopGuard] {self.release_after} "
                            "consecutive clean latched audits; releasing "
                            "the full-budget latch (next eligible proposal "
                            "may stop normally again)",
                            flush=True,
                        )
                else:
                    self._clean_streak = 0

    def should_recover_initial(self, initial_coverage, final_coverage):
        """True only for a critical render collapse that optimization caused."""
        if (not self.enabled or self.audit_every == 0
                or len(self._coverage_history) < self.min_history):
            return False
        if not self._finite(initial_coverage) or not self._finite(final_coverage):
            return False
        median = float(np.median(self._coverage_history))
        critical = self.critical_coverage_ratio * median
        recover = (
            float(final_coverage) < critical
            and float(initial_coverage) >= 2.0 * max(float(final_coverage), 1e-12)
        )
        if recover:
            self.recoveries += 1
            self.latched_full_budget = True
            self.latch_reason = "critical-render-collapse"
            # A SECOND catastrophic event, whether this is the frame that
            # first armed the latch or one that happened during an already-
            # latched probation period, voids any release progress: it is
            # direct evidence the underlying instability is still present.
            self._clean_streak = 0
            print(
                "[ClosedLoopStopGuard] committed candidate lost render "
                "coverage; recovering the frame's motion prediction and "
                "latching full-budget tracking",
                flush=True,
            )
        return recover

    @property
    def frame_action(self):
        return self._frame_action

    @property
    def frame_audit(self):
        return self._frame_audit

    def summary(self):
        if not self.enabled:
            return "Closed-loop stop guard: disabled"
        if self._audit_motion:
            motion = np.asarray(self._audit_motion, dtype=np.float64)
            loss = np.asarray(self._audit_loss, dtype=np.float64)
            audit = (
                f", audit regret motion p90/max="
                f"{np.percentile(motion, 90):.3g}/{np.max(motion):.3g}, "
                f"loss p90/max={100 * np.percentile(loss, 90):.3g}/"
                f"{100 * np.max(loss):.3g}%"
            )
        else:
            audit = ", audit regret unavailable"
        mode = "calibration-only" if self.audit_every == 0 else "active"
        return (
            f"Closed-loop stop guard: {mode}, "
            f"calibration={self.calibration_frames}, audit_every="
            f"{self.audit_every} eligible proposals, history="
            f"{len(self._coverage_history)}/{self.history_size}; proposals="
            f"{self.proposals}, active_stops={self.active_stops}, audits "
            f"calibration/periodic/latched={self.calibration_audits}/"
            f"{self.periodic_audits}/{self.latched_audits}, veto coverage/motion="
            f"{self.coverage_vetoes}/{self.motion_vetoes}, recoveries="
            f"{self.recoveries}, releases={self.releases}"
            + (f" (release_after={self.release_after})"
               if self.release_after else "")
            + f", latched={self.latched_full_budget}"
            f"({self.latch_reason}){audit}"
        )
