"""Rolling first-excursion recorder for pose-preconditioner diagnostics.

The recorder is deliberately independent of the optimiser.  The caller feeds
it JSON-serialisable per-iteration snapshots; it retains a bounded history and
writes only the first suspicious event plus a short post-trigger tail.  Normal
runs do not construct it, so this diagnostic cannot change their arithmetic.
"""

from __future__ import annotations

from collections import deque
import json
import os
from typing import Any


class FirstExcursionRecorder:
    """Keep the iterations immediately surrounding the first excursion."""

    def __init__(self, path: str, history: int = 128, post: int = 24,
                 raw_step_ratio: float = 4.0,
                 seen_drop_ratio: float = 0.20,
                 loss_spike_ratio: float = 0.0,
                 trigger_frame: int = -1,
                 trigger_iteration: int = 0):
        if not path:
            raise ValueError("an excursion trace path is required")
        if history < 1 or post < 0:
            raise ValueError("history must be >= 1 and post must be >= 0")
        self.path = os.path.abspath(path)
        self.history = int(history)
        self.post = int(post)
        self.raw_step_ratio = float(raw_step_ratio)
        self.seen_drop_ratio = float(seen_drop_ratio)
        self.loss_spike_ratio = float(loss_spike_ratio)
        self.trigger_frame = int(trigger_frame)
        self.trigger_iteration = int(trigger_iteration)
        self._records: deque[dict[str, Any]] = deque(maxlen=self.history)
        self._trigger: dict[str, Any] | None = None
        self._post_left = 0
        self._written = False
        self._frame = None
        self._frame_seen_max = 0
        self._frame_best_loss = float("inf")

    @property
    def triggered(self) -> bool:
        return self._trigger is not None

    @property
    def written(self) -> bool:
        return self._written

    def _reasons(self, record: dict[str, Any]) -> list[str]:
        reasons = []
        if (self.trigger_frame >= 0
                and int(record.get("frame", -1)) == self.trigger_frame
                and int(record.get("iteration", -1)) == self.trigger_iteration):
            reasons.append("scheduled")
        state = record.get("preconditioner", {})
        if float(state.get("live", 1.0)) <= 0.0:
            reasons.append("empty_gradient")
        ratio = float(state.get("raw_step_ratio", 0.0))
        if self.raw_step_ratio > 0.0 and ratio >= self.raw_step_ratio:
            reasons.append("raw_step_ratio")

        seen = record.get("seen")
        if seen is not None:
            seen = int(seen)
            if (self.seen_drop_ratio > 0.0 and self._frame_seen_max > 0
                    and int(record.get("iteration", 0)) > 0
                    and seen <= self.seen_drop_ratio * self._frame_seen_max):
                reasons.append("visibility_drop")

        loss = record.get("loss")
        if (loss is not None and self.loss_spike_ratio > 1.0
                and self._frame_best_loss < float("inf")
                and float(loss) >= self.loss_spike_ratio * self._frame_best_loss):
            reasons.append("loss_spike")
        return reasons

    def record(self, record: dict[str, Any]) -> None:
        """Add one iteration and flush after the first trigger's tail."""
        if self._written:
            return
        frame = int(record["frame"])
        if frame != self._frame:
            self._frame = frame
            self._frame_seen_max = 0
            self._frame_best_loss = float("inf")

        reasons = self._reasons(record)
        self._records.append(record)

        seen = record.get("seen")
        if seen is not None:
            self._frame_seen_max = max(self._frame_seen_max, int(seen))
        loss = record.get("loss")
        if loss is not None:
            self._frame_best_loss = min(self._frame_best_loss, float(loss))

        if self._trigger is None and reasons:
            self._trigger = {
                "frame": frame,
                "iteration": int(record["iteration"]),
                "reasons": reasons,
            }
            self._post_left = self.post
            if self._post_left == 0:
                self._write()
        elif self._trigger is not None:
            self._post_left -= 1
            if self._post_left <= 0:
                self._write()

    def close(self) -> None:
        """Flush a triggered trace whose requested tail did not complete."""
        if self._trigger is not None and not self._written:
            self._write()

    def _write(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        meta = {
            "type": "preconditioner_first_excursion",
            "trigger": self._trigger,
            "history": self.history,
            "post": self.post,
            "raw_step_ratio": self.raw_step_ratio,
            "seen_drop_ratio": self.seen_drop_ratio,
            "loss_spike_ratio": self.loss_spike_ratio,
            "trigger_frame": self.trigger_frame,
            "trigger_iteration": self.trigger_iteration,
            "records": len(self._records),
        }
        with open(self.path, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(meta, sort_keys=True) + "\n")
            for record in self._records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._written = True

    def summary(self) -> str:
        if self._written:
            return (f"excursion trace written to {self.path} at frame "
                    f"{self._trigger['frame']} iteration "
                    f"{self._trigger['iteration']} "
                    f"({','.join(self._trigger['reasons'])})")
        if self._trigger is not None:
            return (f"excursion trace triggered at frame {self._trigger['frame']} "
                    f"iteration {self._trigger['iteration']}; pending tail")
        return "excursion trace saw no trigger"
