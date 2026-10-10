"""Tests for kept_pixel_fraction - the loss/coverage rescale under a tile mask.

The number this returns divides the masked tracking loss and the guard's
coverage reading. Getting it wrong does not raise; it silently biases the
best-loss comparison and the guard's collapse latch, which is how the GSLAM
sparse arm diverged to a 234 m ATE while reporting a healthy-looking mask.

CPU-only by design: tile_dims() falls back to 16x16 without the rasterizer
extension, so this runs anywhere.
"""
import torch

from pixel_sample import kept_pixel_fraction, tile_dims


def _grid(H, W):
    bx, by = tile_dims()
    return (H + by - 1) // by, (W + bx - 1) // bx


def test_all_true_is_one_when_divisible():
    H, W = 480, 640
    th, tw = _grid(H, W)
    m = torch.ones(th * tw, dtype=torch.bool)
    assert abs(float(kept_pixel_fraction(m, H, W)) - 1.0) < 1e-6


def test_all_false_is_zero():
    H, W = 480, 640
    th, tw = _grid(H, W)
    m = torch.zeros(th * tw, dtype=torch.bool)
    assert float(kept_pixel_fraction(m, H, W)) == 0.0


def test_half_of_a_divisible_grid():
    H, W = 480, 640
    th, tw = _grid(H, W)
    m = torch.zeros(th * tw, dtype=torch.bool)
    m[: (th * tw) // 2] = True
    assert abs(float(kept_pixel_fraction(m, H, W)) - 0.5) < 1e-6


def test_all_true_is_one_when_not_divisible():
    """The whole point of weighting: padded tiles are not full tiles.

    A plain tile_mask.float().mean() also returns 1.0 here, so this alone
    does not separate the two - see the next test, which does.
    """
    H, W = 100, 100
    th, tw = _grid(H, W)
    m = torch.ones(th * tw, dtype=torch.bool)
    assert abs(float(kept_pixel_fraction(m, H, W)) - 1.0) < 1e-6


def test_padded_edge_tiles_count_less_than_full_ones():
    """100x100 on a 16x16 grid is 7x7 tiles; the last row is only 4px tall.

    Keeping just that row is 7/49 = 14.3% of TILES but 400/10000 = 4% of
    PIXELS. The naive tile mean overstates the kept fraction by 3.6x, which
    would under-correct the masked loss by the same factor.
    """
    H, W = 100, 100
    th, tw = _grid(H, W)
    m = torch.zeros(th, tw, dtype=torch.bool)
    m[-1, :] = True
    got = float(kept_pixel_fraction(m.reshape(-1), H, W))
    assert abs(got - 0.04) < 1e-6, got
    naive = float(m.reshape(-1).to(torch.float32).mean())
    assert abs(naive - 7.0 / 49.0) < 1e-6
    assert got < naive


def test_reciprocal_restores_a_masked_sum_to_the_dense_scale():
    """The property the tracker actually relies on.

    A uniform image summed over a masked render, scaled by 1/fraction, must
    recover the dense sum - that is what makes a dense candidate and a masked
    candidate comparable in the best-loss selection.
    """
    H, W = 480, 640
    bx, by = tile_dims()
    th, tw = _grid(H, W)
    m = torch.zeros(th * tw, dtype=torch.bool)
    m[: int(0.75 * th * tw)] = True

    per_pixel = torch.ones(H, W)
    tile_keep = m.reshape(th, tw)
    pixel_keep = tile_keep.repeat_interleave(by, 0).repeat_interleave(bx, 1)[:H, :W]

    masked_sum = float((per_pixel * pixel_keep).sum())
    dense_sum = float(per_pixel.sum())
    scale = 1.0 / float(kept_pixel_fraction(m, H, W))
    assert abs(masked_sum * scale - dense_sum) < 1e-3


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
