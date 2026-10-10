import torch


class ESSignalBuffer:
    """Collects the per-iteration tracking signals on-device so the host reads
    them once every K iterations instead of three times every iteration.

    WHY THIS EXISTS. With the iteration graph on, a tracking iteration is one
    cudaGraphLaunch instead of ~230 cudaLaunchKernel calls. The CPU should then
    run far ahead of the GPU. It does not, because the loop immediately blocks
    on three device-to-host syncs per iteration, all of them outside the
    captured region:

        pose_delta_norm ... .item()      feeds early stopping
        loss.item()                      feeds early stopping
        loss < current_min_loss          feeds candidate-pose selection

    A sync drains the queue, so the collapsed dispatch is handed straight back.
    That is the most likely reason the graph took 12.1ms/iter to 10.0 rather
    than to the ~6ms of measured GPU work.

    WHAT IT DOES. Every signal the host needs is written into a fixed-address
    ring buffer from INSIDE the captured region, and drained with a single
    .cpu() every K iterations:

        col 0     loss
        col 1     ||pose delta|| over this iteration
        col 2:6   cam_unnorm_rot after the optimizer step
        col 6:9   cam_tran       after the optimizer step

    Rows 2:9 are what makes the candidate-pose selection batchable: with the
    pose of every iteration on hand at drain time, `argmin loss` over the batch
    is decided on the host in plain Python, with exactly the arithmetic the
    eager `if loss < current_min_loss` path uses. That is deliberately NOT the
    approach `sync_free_candidates` took (torch.where on-device, ATE 64.28cm,
    root cause never found); the comparison stays on the host, only its
    FREQUENCY changes.

    WHAT IT COSTS. A stop decision is seen up to K-1 iterations late, so a
    frame may run a few iterations past where it would have stopped. Those
    iterations cannot damage the pose: the committed pose is the best-loss
    candidate, and extra iterations can only add more candidates. Against
    min_iters=70 and an average fire at iteration 94, K=8 is under 8% of one
    frame's tail and no accuracy risk at all - only a little wasted time on
    frames that fire.

    CAPTURE SAFETY. Everything written here has a static shape and a fixed
    address: `_buf` and `_idx` are allocated once per RUN, never per frame, so
    they live outside the graph's private pool and the recorded kernels always
    address the same memory. The slot index has to come from device memory -
    replay re-executes one recorded instruction, so a Python-side `iter % K`
    would be frozen at its capture-time value - hence index_copy_ with a
    device counter rather than a slice assignment.

    Per-frame usage:
        sig.reset_for_frame()
        for it in ...:
            def iteration_fn():
                sig.record_pre_step(rot, tran)   # before loss/backward/step
                ...loss, backward, optimizer.step()...
                sig.record_post_step(loss, rot, tran)
                return ...
            ...graph.run(iteration_fn)...
            for row in sig.drain_if_full():
                ...feed stopper, update candidate...
        for row in sig.drain_remainder():
            ...same...
    """

    # loss, pose_delta, rot(4), tran(3)
    ROW = 9

    def __init__(self, cfg: dict, device="cuda", dtype=torch.float32,
                 pose_dim=7, extra_cols=0, include_pose_pair=False):
        """
        pose_dim: width of the pose snapshot. 7 for SplaTAM (quaternion + xyz).
            MonoGS optimises a 6-vector (cam_rot_delta + cam_trans_delta), so it
            passes 6 and the rot/tran split below follows.
        extra_cols: additional scalars per row, appended after the pose. MonoGS
            uses one for update_pose's `converged` flag, which is a device
            tensor that the eager loop reads with a Python `if` every iteration.
            SplaTAM passes 0 and its row layout is unchanged.
        include_pose_pair: append the exact pre-step and post-step poses. This
            lets SplaTAM recover a gauge-free SE(3) tangent after the batched
            device-to-host drain instead of treating quaternion scale drift as
            camera motion. False preserves every existing row layout.
        """
        self.enabled = bool(cfg.get("enabled", False))
        self.batch = max(1, int(cfg.get("batch", 8)))
        # COMMIT_AT_LOSS. Write the pose the loss was EVALUATED at, not the one
        # the optimiser stepped to afterwards. See record_post_step.
        self.pose_at_loss = bool(cfg.get("pose_at_loss", False))

        self._device = device
        self._dtype = dtype
        self._pose_dim = int(pose_dim)
        self._extra_cols = int(extra_cols)
        self._include_pose_pair = bool(include_pose_pair)
        self._pair_cols = 2 * self._pose_dim if self._include_pose_pair else 0
        self._row = (2 + self._pose_dim + self._pair_cols
                     + self._extra_cols)
        self._extra_start = 2 + self._pose_dim + self._pair_cols
        self._buf = None
        self._idx = None
        self._prev_pose = None
        self._extra_default = None
        self._host_idx = 0       # mirrors the device ring index without a read
        self._count = 0          # iterations recorded this frame
        self._drained = 0        # iterations already handed to the host
        if self.enabled:
            # Allocated once for the whole run. Reallocating per frame would
            # move the addresses the captured kernels were recorded against.
            self._buf = torch.zeros(self.batch, self._row, device=device, dtype=dtype)
            self._idx = torch.zeros(1, dtype=torch.long, device=device)
            self._prev_pose = torch.zeros(self._pose_dim, device=device, dtype=dtype)
            if self._extra_cols:
                self._extra_default = torch.full(
                    (self._extra_cols,), float("nan"),
                    device=device, dtype=dtype)

        self.drains = 0
        self.iterations = 0

        print(f"[ESSignalBuffer] constructed (enabled={self.enabled}, "
              f"batch={self.batch}, pose_dim={self._pose_dim}, "
              f"extra_cols={self._extra_cols}, "
              f"pose_pair={self._include_pose_pair})", flush=True)

    # ── frame lifecycle ───────────────────────────────────────────────────────

    def reset_for_frame(self):
        if not self.enabled:
            return
        self._idx.zero_()
        self._buf.zero_()
        self._count = 0
        self._drained = 0
        self._host_idx = 0

    # ── inside the captured region ────────────────────────────────────────────

    def record_pre_step(self, rot, tran):
        """Snapshot the pose before the optimizer step. Device-only."""
        if not self.enabled:
            return
        self._prev_pose.copy_(
            torch.cat([rot.detach().reshape(-1), tran.detach().reshape(-1)])
        )

    def record_post_step(self, loss, rot, tran, extra=None):
        """Write this iteration's row. Device-only, static shape, fixed address.

        extra: iterable of 0-dim/1-element device tensors, exactly extra_cols of
        them. Booleans are fine - they are cast to the buffer dtype, so the
        drain returns 0.0/1.0 and the caller compares against 0.5.
        """
        if not self.enabled:
            return
        now = torch.cat([rot.detach().reshape(-1), tran.detach().reshape(-1)])
        delta = (now - self._prev_pose).norm().reshape(1)
        # THE ROW PAIRS A LOSS WITH A POSE, AND THEY MUST BE THE SAME MOMENT.
        #
        # `loss` is computed at the TOP of the iteration - before backward,
        # before the step - so it describes _prev_pose. `now` is read after the
        # step. Writing them together means the drain's `argmin loss` selects a
        # row whose pose is ONE OPTIMISER STEP past the one that actually
        # achieved that loss, and the size of the error is the size of the last
        # step.
        #
        # That is why it surfaces as map quality rather than trajectory error:
        # ATE is aligned over the whole run so a per-frame offset partly
        # cancels, while every rendered frame carries the full misalignment.
        # Measured on TUM fr1 full length, handoff (Adam tail, 0.49x ref) vs
        # handoff-free (0.04-0.08x): ATE 3.86 vs 3.38, PSNR 19.00 vs 21.53.
        #
        # `delta` is unaffected either way - it is the distance between the two
        # poses, and both are still in hand.
        #
        # Default OFF: this changes every committed pose in every arm and both
        # ladders' numbers were recorded with the current behaviour. Add the
        # flag, measure, then flip.
        parts = [
            loss.detach().reshape(1).to(self._dtype),
            delta.to(self._dtype),
            (self._prev_pose if self.pose_at_loss else now).to(self._dtype),
        ]
        if self._include_pose_pair:
            parts.extend([self._prev_pose.to(self._dtype), now.to(self._dtype)])
        if self._extra_cols:
            if extra is None:
                parts.append(self._extra_default)
            else:
                parts.extend(
                    e.detach().reshape(1).to(self._dtype) for e in extra)
        row = torch.cat(parts).unsqueeze(0)
        self._buf.index_copy_(0, self._idx, row)
        # The slot must advance on the device: replay re-runs the recorded
        # instruction, so anything computed on the host at capture time would
        # be a constant for the rest of the frame.
        self._idx.add_(1).remainder_(self.batch)

    # ── outside the captured region ───────────────────────────────────────────

    def note_iteration(self):
        """Call once per iteration, after the iteration has been issued."""
        if not self.enabled:
            return
        self._count += 1
        self.iterations += 1
        self._host_idx = (self._host_idx + 1) % self.batch

    def write_current_extra(self, extra):
        """Overwrite the current row's extra columns without a host read.

        This is called after the captured iteration has written its default
        NaNs but before note_iteration() advances the mirrored host slot. It
        lets a GPU scalar computed immediately after graph replay ride along
        with the next drain instead of opening a separate synchronization.
        """
        if not self.enabled:
            return
        if not self._extra_cols:
            raise RuntimeError("write_current_extra requires extra_cols > 0")
        value = extra.detach().reshape(-1)
        if value.numel() != self._extra_cols:
            raise ValueError(
                "expected %d extra values, got %d"
                % (self._extra_cols, value.numel()))
        self._buf[self._host_idx, self._extra_start:].copy_(
            value.to(device=self._device, dtype=self._dtype))

    def skip_iteration(self):
        """Advance the iteration index without claiming that a row exists.

        Adaptive proposal validation renders and backpropagates once, but a
        rejected full-matrix proposal is rolled back and has no honest
        loss/step row to feed convergence. Call this only after drain_phase(),
        when there are no pending rows; advancing both counters preserves the
        real zero-based iteration number of every later row.
        """
        if not self.enabled:
            return
        if self._count != self._drained:
            raise RuntimeError(
                "skip_iteration requires all pending signal rows to be drained")
        self._count += 1
        self._drained += 1
        self.iterations += 1

    def drain_if_full(self):
        """Return the pending rows if the ring just filled, else []. One sync."""
        if not self.enabled or self._count == self._drained:
            return []
        if (self._count - self._drained) < self.batch:
            return []
        return self._drain(self.batch)

    def drain_remainder(self):
        """Return whatever is still pending at the end of a frame. One sync."""
        if not self.enabled:
            return []
        n = self._count - self._drained
        if n <= 0:
            return []
        return self._drain(n)

    def drain_phase(self):
        """Drain a partial batch and restart ring order at a phase boundary.

        ``_drain`` normally sees either a full ring (write index back at zero)
        or the final partial ring of a frame. A mid-frame objective/optimizer
        transition is different: after draining its partial prefix, later
        writes must restart at slot zero or the next full drain would return
        the ring in rotated rather than chronological order.
        """
        if not self.enabled:
            return []
        rows = self.drain_remainder()
        self._idx.zero_()
        self._buf.zero_()
        self._host_idx = 0
        return rows

    def _drain(self, n):
        # Rows 0..n-1 hold the oldest-first pending iterations: the ring is
        # drained the moment it fills, so the write index is either back at 0
        # (full drain) or at n (partial, end of frame).
        host = self._buf[:n].to("cpu", non_blocking=False)
        self.drains += 1
        base = self._drained
        self._drained += n
        out = []
        rot_end = 2 + (4 if self._pose_dim == 7 else 3)
        pose_end = 2 + self._pose_dim
        for j in range(n):
            r = host[j]
            row = {
                "iter":       base + j,
                "loss":       float(r[0]),
                "pose_delta": float(r[1]),
                "rot":        r[2:rot_end],
                "tran":       r[rot_end:pose_end],
            }
            extra_start = pose_end
            if self._include_pose_pair:
                pre_start = pose_end
                pre_end = pre_start + self._pose_dim
                post_end = pre_end + self._pose_dim
                pair_rot_width = 4 if self._pose_dim == 7 else 3
                row.update({
                    "pre_rot": r[pre_start:pre_start + pair_rot_width],
                    "pre_tran": r[pre_start + pair_rot_width:pre_end],
                    "post_rot": r[pre_end:pre_end + pair_rot_width],
                    "post_tran": r[pre_end + pair_rot_width:post_end],
                })
                extra_start = post_end
            if self._extra_cols:
                row["extra"] = [
                    float(r[extra_start + k]) for k in range(self._extra_cols)
                ]
            out.append(row)
        return out

    # ── reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        if not self.enabled:
            return "Early-stop signal batching: disabled"
        per_iter = self.drains / self.iterations if self.iterations else 0.0
        return (f"Early-stop signal batching: batch={self.batch}, "
                f"{self.drains} drains over {self.iterations} iterations "
                f"({per_iter:.3f} syncs/iter, eager path is 3.000)")
