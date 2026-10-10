"""
grad_needs.py - which gradients does THIS render actually need?

Tracking and mapping optimise different things, so they need different
gradients out of the same rasterizer. The existing `tracking_only` flag is the
first instance of exploiting that; this module is where the per-render decision
rules live, so all three models share one implementation and one set of
correctness arguments instead of re-deriving them at each call site.

WHY THIS MATTERS NOW. A ceiling probe (DGR_STUB_GRADS) measured the
per-Gaussian gradient accumulation at **52.4% of renderCUDABackward** on
SplaTAM - by far the largest single cost identified on this branch. Anything
provably dead that can be removed from it is worth removing, and unlike a
reduction rewrite it costs no accuracy.

----------------------------------------------------------------------------
  colour_grad_is_dead()
----------------------------------------------------------------------------

`dL_dcolors` is accumulated with one atomicAdd per channel per contribution -
3 of the ~9 atomics in the common path. It is DEAD when the pose cannot reach
the colours, which is exactly when all three of these hold:

  1. this is a TRACKING render (mapping optimises colours directly)
  2. colours are PRECOMPUTED (on the SH path the pose reaches colour through
     the view direction, and dL_dcolors feeds dL_dmean via
     computeColorFromSH - killing it there would corrupt the pose gradient)
  3. the colour tensor is a LEAF, i.e. grad_fn is None

CONDITION 3 IS THE ONE THAT DOES THE REAL WORK, and it is checkable rather
than assumed. A leaf tensor was not computed from anything, so it cannot be a
function of the pose. A tensor WITH a grad_fn was computed, possibly from the
pose - and in SplaTAM it demonstrably is:

    RGB render          colors_precomp = params['rgb_colors']
                        a leaf nn.Parameter, frozen during tracking
                        -> grad_fn is None  -> DEAD, safe to skip

    depth+sil render    colors_precomp = get_depth_and_silhouette(...)
                        computed from the POSE-TRANSFORMED means
                        -> grad_fn is not None -> LIVE, must not skip

Those two calls sit three lines apart and differ only in this. A global
"tracking" flag would wrongly cover both; the leaf check separates them
automatically, on evidence rather than on a comment.

DEFAULTS TO FALSE on anything it cannot prove. An unnecessary atomic costs
time; a missing gradient costs correctness.

----------------------------------------------------------------------------
  PER MODEL
----------------------------------------------------------------------------

  SplaTAM         RGB render qualifies; depth/silhouette render does not.
  Gaussian-SLAM   depends on whether its render passes precomputed colours or
                  SH - the rule decides per call, no per-model special case.
  MonoGS          uses SH, so condition 2 fails and this always returns False.
                  Its `tracking_only` already skips dL_dsh, which is the
                  separate and legitimate saving on that path.

USAGE

    from utils.grad_needs import colour_grad_is_dead
    skip = colour_grad_is_dead(colors_precomp, tracking=True)
    ... = Renderer(...)(..., skip_color_grad=skip)
"""

from __future__ import annotations


def colour_grad_is_dead(colors_precomp, tracking: bool) -> bool:
    """True only when dL_dcolors is PROVABLY unused for this render.

    See the module docstring for the argument. Conservative by construction:
    every branch that cannot establish deadness returns False.
    """
    if not tracking:
        # Mapping optimises the colours themselves.
        return False
    if colors_precomp is None:
        # SH path: the pose reaches colour through the view direction, and
        # dL_dcolors feeds dL_dmean via computeColorFromSH.
        return False
    if getattr(colors_precomp, "grad_fn", None) is not None:
        # Computed from something - possibly the pose. SplaTAM's depth and
        # silhouette channels are exactly this case.
        return False
    if not getattr(colors_precomp, "is_leaf", True):
        # Belt and braces: a non-leaf with no grad_fn should not occur, but
        # deadness is not established, so do not claim it.
        return False
    return True


def explain(colors_precomp, tracking: bool) -> str:
    """One line saying WHY, for a log or a smoke test.

    The decision is invisible in a timing run - a skipped atomic looks like
    nothing - so a run that wants to confirm the path engaged can print this.
    """
    if not tracking:
        return "colour grad LIVE: mapping render, colours are being optimised"
    if colors_precomp is None:
        return "colour grad LIVE: SH path, pose reaches colour via view direction"
    if getattr(colors_precomp, "grad_fn", None) is not None:
        return (f"colour grad LIVE: colours have grad_fn "
                f"({type(colors_precomp.grad_fn).__name__}), so they may depend "
                f"on the pose")
    return "colour grad DEAD: tracking render, colours are a frozen leaf tensor"
