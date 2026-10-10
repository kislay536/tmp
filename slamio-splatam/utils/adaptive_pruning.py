import torch


class AdaptivePruner:
    """Model-agnostic adaptive Gaussian pruning decision logic.

    Deliberately does not use RTGS's gradient-informed ranking
    (xyz_gradient_accum/denom) - that signal is only populated in models/
    configs where splatting-based densification is already running, and
    porting it just for this would mean turning on unrelated always-on
    accumulation. Ranks on opacity/age instead, both of which every model
    variant here already tracks unconditionally: opacity is a core
    3DGS parameter, and "age" needs only a per-Gaussian creation-frame
    counter, not a full gradient-accumulation pass.

    Usage per prune decision (once per frame, not per mapping iteration):
        pruner = AdaptivePruner(config.get("adaptive_pruning", {}))
        mask = pruner.compute_mask(opacity, age, progress)
        # mask is a bool tensor, True = prune. Caller owns the actual
        # removal mechanics (its own optimizer state, parameter dict).
        print(pruner.summary())
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        # Fraction of the way through the sequence (frame_idx / num_frames)
        # before pruning activates at all - protects the whole early part
        # of the run, where the map is still forming and most Gaussians
        # are "new" by definition.
        self.start_progress = float(cfg.get("start_progress", 0.75))
        # Per-call cap as a fraction of the current Gaussian count - a
        # rate limit, not a one-shot cutoff, so a single frame's removal
        # can't cause a large allocator/optimizer-state churn spike.
        self.max_prune_fraction_per_step = float(cfg.get("max_prune_fraction_per_step", 0.03))
        # Gaussians younger than this (frames since creation) are never
        # pruned, regardless of opacity - too new to judge.
        self.min_age_protect = int(cfg.get("min_age_protect", 20))
        # Below this opacity, a Gaussian is a pruning candidate.
        self.opacity_threshold = float(cfg.get("opacity_threshold", 0.05))
        # At or above this opacity, a Gaussian is protected regardless of
        # age - confident, established Gaussians are never candidates.
        # Kept as a distinct knob from opacity_threshold (not simply
        # "not below it") so the two can be tuned independently: e.g. a
        # dead zone between them where age is the deciding factor.
        self.min_opacity_protect = float(cfg.get("min_opacity_protect", 0.85))

        self._total_calls = 0
        self._total_seen = 0
        self._total_pruned = 0

        print(f"[AdaptivePruner] constructed (enabled={self.enabled}, "
              f"start_progress={self.start_progress}, "
              f"max_prune_fraction_per_step={self.max_prune_fraction_per_step}, "
              f"min_age_protect={self.min_age_protect}, "
              f"opacity_threshold={self.opacity_threshold})", flush=True)

    def compute_mask(self, opacity: torch.Tensor, age: torch.Tensor, progress: float,
                      protected: torch.Tensor = None) -> torch.Tensor:
        """Return a bool tensor (True = prune) over the current Gaussians.

        opacity: per-Gaussian opacity in [0, 1], same length as age.
        age: per-Gaussian age in frames (current_frame_idx - creation_frame).
        progress: float in [0, 1], how far through the sequence this call is.
        protected: optional bool tensor, True = never prune this Gaussian
            regardless of the other criteria (e.g. active keyframe window).
        """
        n_total = opacity.shape[0]
        if not self.enabled or progress < self.start_progress:
            return torch.zeros(n_total, dtype=torch.bool, device=opacity.device)

        candidate = (opacity < self.opacity_threshold) & (age >= self.min_age_protect)
        candidate = candidate & (opacity < self.min_opacity_protect)
        if protected is not None:
            candidate = candidate & (~protected)

        max_remove = int(self.max_prune_fraction_per_step * n_total)
        n_candidates = int(candidate.sum().item())
        if max_remove <= 0:
            candidate = torch.zeros_like(candidate)
        elif n_candidates > max_remove:
            # Keep only the lowest-opacity `max_remove` candidates - the
            # rate cap, applied by severity rather than an arbitrary cutoff.
            idx = candidate.nonzero(as_tuple=True)[0]
            _, order = torch.sort(opacity[idx])
            keep_idx = idx[order[:max_remove]]
            new_candidate = torch.zeros_like(candidate)
            new_candidate[keep_idx] = True
            candidate = new_candidate

        self._total_calls += 1
        self._total_seen += n_total
        self._total_pruned += int(candidate.sum().item())
        return candidate

    def summary(self) -> str:
        if not self.enabled:
            return "Adaptive pruning: disabled"
        avg_frac = (100 * self._total_pruned / self._total_seen) if self._total_seen else 0
        return (f"Adaptive pruning: {self._total_pruned} Gaussians removed over "
                f"{self._total_calls} calls (avg {avg_frac:.2f}% of live count per call)")
