import torch
import torch.nn.functional as F

_TILE_DIMS = None


def tile_dims():
    """(BLOCK_X, BLOCK_Y) of the INSTALLED rasterizer, not a hardcoded 16.

    The tile size is a build variant now (DGR_BLOCK_X / DGR_BLOCK_Y), and the
    mask built here is indexed by tile id in the rasterizer's grid. If the two
    disagree the mask is simply the wrong length for the grid and NOTHING
    RAISES - tiles get masked at random offsets. Read it from the extension.

    Falls back to 16x16 for a .so predating the introspection binding, which
    genuinely was 16x16.
    """
    global _TILE_DIMS
    if _TILE_DIMS is None:
        try:
            from diff_gaussian_rasterization import tile_size
            _TILE_DIMS = tile_size()
        except Exception:
            _TILE_DIMS = (16, 16)
    return _TILE_DIMS


# Back-compat for any caller still reading the old module constant. Square
# builds keep the old meaning; a non-square build makes this ambiguous, so use
# tile_dims() instead.
TILE_SIZE = 16


def build_tile_mask(gt_image, H, W, cfg):
    """
    Returns a (tile_h * tile_w,) bool CUDA tensor.
    True  = render this tile normally.
    False = skip tile (forward writes background; backward skips via n_contrib=0).

    Keeps the top `sample_ratio` fraction of tiles ranked by mean image-gradient
    magnitude (edges/texture tiles are most informative for pose optimisation).
    """
    sample_ratio  = cfg.get('sample_ratio', 0.6)
    gradient_frac = cfg.get('gradient_frac', 0.5)

    bx, by = tile_dims()

    # CEIL, matching rasterizer_impl.cu's grid:
    #     grid.x = (W + BLOCK_X - 1) / BLOCK_X,  grid.y = (H + BLOCK_Y - 1) / BLOCK_Y
    # This used floor, so any resolution not divisible by the tile size built a
    # mask SHORTER than the grid - silently, since nothing checks the length.
    # TUM at 640x480 divides exactly by 8, 16 and 32 so it never bit, but a
    # tile-size sweep is exactly when it would.
    tile_h = (H + by - 1) // by
    tile_w = (W + bx - 1) // bx

    # gt_image: (3, H, W) on CUDA, values in [0, 1]
    gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]  # (H, W)

    dx = (gray[:, 1:] - gray[:, :-1]).abs()
    dy = (gray[1:, :] - gray[:-1, :]).abs()
    dx = torch.cat([dx, dx[:, -1:]], dim=1)
    dy = torch.cat([dy, dy[-1:, :]], dim=0)
    grad_mag = dx + dy  # (H, W)

    # Pad up to a whole number of tiles so the reshape is exact. Replicate
    # rather than zero-pad: a zero border would read as a flat, uninformative
    # tile and bias the edge tiles out of the top-gradient selection.
    pad_h = tile_h * by - H
    pad_w = tile_w * bx - W
    if pad_h or pad_w:
        grad_mag = F.pad(grad_mag[None, None], (0, pad_w, 0, pad_h),
                         mode="replicate")[0, 0]

    tile_scores = grad_mag.reshape(tile_h, by, tile_w, bx).mean(dim=(1, 3))
    flat_scores = tile_scores.flatten()  # (tile_h * tile_w,)

    n_tiles = tile_h * tile_w
    n_keep  = max(1, int(round(n_tiles * sample_ratio)))
    n_grad  = max(1, int(round(n_keep * gradient_frac)))
    n_unif  = n_keep - n_grad

    _, top_idx = torch.topk(flat_scores, min(n_grad, n_tiles))
    tile_mask = torch.zeros(n_tiles, dtype=torch.bool, device=gt_image.device)
    tile_mask[top_idx] = True

    if n_unif > 0:
        unselected = (~tile_mask).nonzero(as_tuple=False).squeeze(1)
        if unselected.numel() > 0:
            perm = torch.randperm(unselected.numel(), device=gt_image.device)[:n_unif]
            tile_mask[unselected[perm]] = True

    return tile_mask  # (tile_h * tile_w,) bool on CUDA


def kept_pixel_fraction(tile_mask, H, W):
    """Fraction of the H x W IMAGE that `tile_mask` leaves rendered.

    NOT tile_mask.float().mean(). The tile grid is ceil-padded, so the last
    row and column of tiles hang off the image edge and carry fewer than
    by*bx real pixels. At 640x480 with 16x16 tiles the two agree exactly;
    at any resolution the tile size does not divide they do not, and the
    difference lands straight in the loss scale - which is precisely the
    quantity this is used to correct.

    Returns a 0-dim CUDA tensor so callers can divide by it without a host
    sync, and legally inside a CUDA graph capture.
    """
    bx, by = tile_dims()
    tile_h = (H + by - 1) // by
    tile_w = (W + bx - 1) // bx

    # Built on the host and moved in one copy: `rows[-1] = ...` straight into
    # a CUDA tensor would be a host round trip per frame.
    rows = torch.full((tile_h,), float(by))
    cols = torch.full((tile_w,), float(bx))
    rows[-1] = float(H - by * (tile_h - 1))
    cols[-1] = float(W - bx * (tile_w - 1))
    weights = (rows[:, None] * cols[None, :]).to(tile_mask.device)

    grid = tile_mask.reshape(tile_h, tile_w).to(torch.float32)
    return (grid * weights).sum() / float(H * W)


def is_sparse_phase(iter_idx, num_iters, cfg):
    """Returns True when the current iteration falls in the sparse window."""
    if not cfg.get('enabled', False) or num_iters == 0:
        return False
    frac       = iter_idx / max(num_iters - 1, 1)
    full_start = cfg.get('full_start_ratio', 0.2)
    full_end   = cfg.get('full_end_ratio', 0.8)
    return full_start <= frac < full_end


class PixelSampleTracker:
    """Diagnostic wrapper around build_tile_mask()/is_sparse_phase().

    Those two stay plain functions (shared as-is with MonoGS's own
    wiring) - this only adds the same construction/summary print pattern
    every other opt-tracking class here already has (AdaptiveMapper,
    EarlyStop, AdaptivePruner), so a run's log can confirm whether
    pixel_sample actually engaged instead of guessing from wall-time/ATE
    deltas alone.
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.sample_ratio = float(cfg.get("sample_ratio", 0.6))
        self.full_start_ratio = float(cfg.get("full_start_ratio", 0.2))
        self.full_end_ratio = float(cfg.get("full_end_ratio", 0.8))
        self._sparse_iters = 0
        self._total_iters = 0

        print(f"[PixelSample] constructed (enabled={self.enabled}, "
              f"sample_ratio={self.sample_ratio}, "
              f"full_start_ratio={self.full_start_ratio}, "
              f"full_end_ratio={self.full_end_ratio})", flush=True)

    def note_iter(self, is_sparse: bool):
        self._total_iters += 1
        if is_sparse:
            self._sparse_iters += 1

    def summary(self) -> str:
        if not self.enabled:
            return "Tile-mask pixel sampling: disabled"
        pct = 100 * self._sparse_iters / self._total_iters if self._total_iters else 0
        return (f"Tile-mask pixel sampling: {self._sparse_iters}/{self._total_iters} "
                f"iters masked ({pct:.1f}%)")
