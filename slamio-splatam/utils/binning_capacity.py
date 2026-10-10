import torch


class BinningCapacity:
    """Chooses a fixed binning-buffer capacity for the rasterizer.

    The rasterizer normally reads num_rendered (total tile-Gaussian
    intersections) back to the host to size its binning buffer, the sort and
    one launch grid. That synchronous memcpy is illegal while a CUDA stream
    is capturing, so it is the single thing preventing the rasterizer from
    being recorded into a graph.

    num_rendered cannot be derived from image resolution or Gaussian count -
    it depends on how large each Gaussian's screen-space footprint is, which
    is a function of pose and scene content. Its static worst case is
    P * tile_count, which is uselessly large. But during tracking the pose
    moves only ~6e-4 per iteration, so it is nearly constant within a frame:
    measuring it on the eager warm-up iterations and adding a margin gives a
    bound that holds for the rest of the frame.

    Overflow is still possible, so duplicateWithKeys bounds-checks its writes
    and sets a device flag. That flag is read ONCE PER FRAME, outside any
    captured region - which is the point: ~117 syncs per frame become one.

    Per-frame usage:
        cap.reset_for_frame()
        for it in range(...):
            kwargs = cap.render_kwargs()
            ... render/loss/backward/step ...
            cap.note_iteration()
        cap.end_frame()
    """

    def __init__(self, cfg: dict, device="cuda"):
        self.enabled = bool(cfg.get("enabled", False))
        self.margin = float(cfg.get("margin", 1.25))
        self.warmup_iters = int(cfg.get("warmup_iters", 3))
        # Absolute floor on the capacity, independent of the relative margin.
        #
        # WHY. capacity = observed_max * margin, and observed_max comes from
        # only `warmup_iters` iterations. A relative margin gives sensible
        # headroom when observed_max is large and almost none when it is small:
        # measured on an adaptive-mapping run, frame 346 observed 8578 and got
        # a capacity of 24125, which a few thousand extra intersections blew
        # straight through. The failure then cascades, because AN OVERFLOW
        # MEANS THE RENDER DROPPED INSTANCES - that frame's render is wrong,
        # tracking degrades, and the next frame is worse.
        #
        # A floor costs almost nothing: the binning buffer is ~12 bytes per
        # entry, so 400k entries is under 5MB against a tuned run's typical
        # ~1.1M capacity. Default 0 (off) so no existing measurement moves.
        self.min_capacity = int(cfg.get("min_capacity", 0))
        # Cap the self-raising policy. It multiplies by 1.5 per overflow up to
        # 8.0 and NEVER lowers again, so one bad frame taxes every remaining
        # frame of the run. On the tuned fr1 config this never fired (0
        # overflows in 591 frames) and was therefore never exercised.
        self.max_margin = float(cfg.get("max_margin", 8.0))

        self._device = device
        self._count_out = None
        self._overflow = None
        if self.enabled:
            self._count_out = torch.zeros(1, dtype=torch.int32, device=device)
            self._overflow = torch.zeros(1, dtype=torch.int32, device=device)

        self._iter = 0
        self._observed_max = 0
        self._capacity = -1

        self.frames = 0
        self.overflow_frames = 0
        self._capacity_sum = 0
        self._observed_sum = 0

        print(f"[BinningCapacity] constructed (enabled={self.enabled}, "
              f"margin={self.margin}, warmup_iters={self.warmup_iters})", flush=True)

    def reset_for_frame(self):
        """Call at the start of each frame's tracking."""
        if not self.enabled:
            return
        self._iter = 0
        self._observed_max = 0
        self._capacity = -1
        self._overflow.zero_()
        self.frames += 1

    def render_kwargs(self) -> dict:
        """Extra kwargs for the rasterizer call(s) of the current iteration.

        Both render passes in a tracking iteration rasterize the same
        Gaussians from the same pose, so they share one capacity.
        """
        if not self.enabled:
            return {}
        if self._capacity < 0:
            # Warm-up: run the original readback path and record what it found.
            return {"binning_count_out": self._count_out}
        return {"binning_capacity": self._capacity,
                "binning_overflow": self._overflow}

    def note_iteration(self):
        """Call once per tracking iteration, after the render."""
        if not self.enabled or self._capacity >= 0:
            self._iter += 1
            return
        # int() here syncs, but only on the warm-up iterations - the whole
        # point is that the steady-state iterations do not.
        observed = int(self._count_out.item())
        self._observed_max = max(self._observed_max, observed)
        self._iter += 1
        if self._iter >= self.warmup_iters and self._observed_max > 0:
            self._capacity = max(int(self._observed_max * self.margin),
                                 self.min_capacity)
            self._capacity_sum += self._capacity
            self._observed_sum += self._observed_max

    def end_frame(self) -> bool:
        """Call at the end of a frame. Returns True if capacity overflowed.

        An overflow means duplicateWithKeys dropped instances, so that frame's
        render was wrong. It is reported loudly rather than silently absorbed,
        and the margin is raised so later frames have more headroom.
        """
        if not self.enabled or self._capacity < 0:
            return False
        if int(self._overflow.item()) == 0:
            return False
        self.overflow_frames += 1
        old = self.margin
        self.margin = min(self.margin * 1.5, self.max_margin)
        # The policy is inherently REACTIVE and cannot prevent the first
        # corruption: by the time this runs, the frame has already rendered
        # with dropped instances. Raising the margin only protects later
        # frames. If a config overflows at all, the initial margin was wrong
        # for it - raise that, do not rely on this.
        print(f"[BinningCapacity] OVERFLOW at frame {self.frames}: capacity "
              f"{self._capacity} (observed max {self._observed_max}) was too "
              f"small - this frame's render dropped instances. margin "
              f"{old:.2f} -> {self.margin:.2f}", flush=True)
        if self.overflow_frames == 1:
            print(f"[BinningCapacity] NOTE: the first overflow already produced "
                  f"a wrong render. Margin {old:.2f} is not safe for this "
                  f"configuration - raise binning_capacity.margin, and set "
                  f"min_capacity if observed_max is small.", flush=True)
        return True

    def summary(self) -> str:
        if not self.enabled:
            return "Binning capacity: disabled"
        if self.frames == 0:
            return "Binning capacity: enabled but never used"
        sized = max(self._capacity_sum > 0, 0)
        avg_cap = self._capacity_sum / self.frames if sized else 0
        avg_obs = self._observed_sum / self.frames if sized else 0
        waste = ((avg_cap / avg_obs - 1) * 100) if avg_obs > 0 else 0
        return (f"Binning capacity: {self.frames} frames, avg capacity "
                f"{avg_cap:.0f} vs observed {avg_obs:.0f} (+{waste:.0f}% sorted), "
                f"{self.overflow_frames} overflow frames, final margin "
                f"{self.margin:.2f}")
