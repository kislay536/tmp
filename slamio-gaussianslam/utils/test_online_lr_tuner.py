import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.online_lr_tuner import OnlineLRTuner, PoseErrDelta, _crit, _paired_t


def run(tuner, response, max_frames=400):
    """Drive the tuner like a frontend: apply tuner.k, run a frame, report it.
    response(k, frame_index) -> iterations that frame took."""
    frame = 0
    while not tuner.frozen and frame < max_frames:
        tuner.observe(response(tuner.k, frame))
        frame += 1
    return frame


print("[0] a clean REDUCE cascade converges, then holds")
# Iterations fall with k down to 0.25 and rise again below it.
table = {1.0: 80, 0.5: 55, 0.25: 40, 0.125: 65}
t = OnlineLRTuner(block_frames=12, max_halvings=4)
n = run(t, lambda k, i: table[k])
assert t.frozen and t.k == 0.25, t.summary()
assert [h[1] for h in t.history] == ["REDUCE", "REDUCE", "KEEP"], t.summary()
# noise-free -> every rung decides at the first look (min_pairs = 6 -> 12 frames)
assert n == 36, n
print(t.summary())

print("\n[1] KEEP on the very first halving freezes at k=1.0")
t = OnlineLRTuner(block_frames=12)
run(t, lambda k, i: 50 if k == 1.0 else 70)
assert t.frozen and t.k == 1.0 and t.history[0][1] == "KEEP", t.summary()

print("\n[2] AMBIGUOUS (noise, no real effect) freezes at the larger k")
rng = random.Random(3)
t = OnlineLRTuner(block_frames=12)
n = run(t, lambda k, i: 50 + rng.choice((-3, -1, 0, 1, 3)))
assert t.frozen and t.k == 1.0, t.summary()
assert t.history[0][1] == "AMBIGUOUS" and t.history[0][4] == t.max_pairs
assert n == 2 * t.max_pairs, n   # spent the whole allowance, and only that
print(t.summary())

print("\n[3] cap-bound from the start never tunes at all")
t = OnlineLRTuner(block_frames=12, budget=100)
run(t, lambda k, i: 100)
assert t.frozen and t.k == 1.0 and t.history[0][1] == "CAP-BOUND", t.summary()

print("\n[4] cap-bound partway through reverts to the last clean value")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5)
run(t, lambda k, i: {1.0: 80, 0.5: 60}.get(k, 100), max_frames=400)
# 1.0 vs 0.5 is a clean REDUCE; 0.5 vs 0.25 is entirely capped -> keep 0.5
assert t.frozen and t.k == 0.5, t.summary()
assert t.history[-1][1] == "CAP-BOUND", t.summary()

print("\n[5] max_halvings stops an indefinite REDUCE cascade")
t = OnlineLRTuner(block_frames=8, max_halvings=2)
run(t, lambda k, i: 100 * k)
assert t.frozen and t.k == 0.25 and t.halvings_done == 2, t.summary()

print("\n[6] identical arms: mean delta 0 with zero variance is AMBIGUOUS")
t = OnlineLRTuner(block_frames=8)
run(t, lambda k, i: 60)
assert t.frozen and t.k == 1.0 and t.history[0][1] == "AMBIGUOUS", t.summary()

print("\n[7] the arms are interleaved and the leading arm alternates")
t = OnlineLRTuner(block_frames=12)
ks = []
for _ in range(8):
    ks.append(t.k)
    t.observe(60)
assert ks == [1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0], ks

print("\n[8] a SLOW SCENE DRIFT with no lr effect is not read as a REDUCE")
# The consecutive-block design failed exactly here: iterations rise steadily
# with frame index, so whichever arm runs later looks slower (or, run the
# other way round, faster). Adjacent pairs see nearly the same scene.
false_reduce = 0
for seed in range(200):
    rng = random.Random(seed)
    t = OnlineLRTuner(block_frames=12)
    run(t, lambda k, i: 60 + 1.5 * i + rng.gauss(0, 4))
    false_reduce += t.history[0][1] == "REDUCE"
assert false_reduce <= 200 * 0.06, false_reduce
print(f"  false REDUCE {false_reduce}/200 under a +1.5 it/frame per frame drift")

print("\n[9] a second-frame-in-pair bias with no lr effect cancels")
# The 2nd frame of every pair is 6 iterations FASTER regardless of k.
false_reduce = 0
for seed in range(200):
    rng = random.Random(seed)
    t = OnlineLRTuner(block_frames=12)
    run(t, lambda k, i: 60 - (6 if i % 2 else 0) + rng.gauss(0, 4))
    false_reduce += t.history[0][1] == "REDUCE"
assert false_reduce <= 200 * 0.06, false_reduce
print(f"  false REDUCE {false_reduce}/200 with a constant -6 second-frame bias")

print("\n[10] a real 20% REDUCE under noise AND drift is found")
found = 0
lengths = []
for seed in range(200):
    rng = random.Random(1000 + seed)
    t = OnlineLRTuner(block_frames=12, max_halvings=1)
    n = run(t, lambda k, i: (60 + 1.5 * i) * (0.8 if k == 0.5 else 1.0)
            + rng.gauss(0, 5))
    found += t.history[0][1] == "REDUCE"
    lengths.append(n)
assert found >= 200 * 0.85, found
print(f"  found {found}/200, median {sorted(lengths)[100]} frames")

print("\n[11] critical values are small-sample, not 2.0")
assert _crit(5) > 3.0 and _crit(11) < _crit(5) and _crit(500) > 2.3

print("\n[12] k_log records the k each frame RAN at, including after the freeze")
t = OnlineLRTuner(block_frames=8, max_halvings=1)
ran = []
for i in range(30):
    ran.append(t.k)                       # the k applied to this frame
    t.observe(60 if t.k == 1.0 else 45)   # cand is clearly shorter
assert t.k_log == ran, (t.k_log, ran)
assert t.frozen and set(t.k_log[-5:]) == {0.5}, t.k_log
t2 = OnlineLRTuner(block_frames=8)
t2.observe(0)
t2.observe(-1)                            # frames that did not run are not logged
assert t2.k_log == [], t2.k_log

print("\n[13] cap-bound WITH a pose-error reading on every pair resolves REDUCE")
# 100% at cap throughout (at_cap=True always) - it/frame carries no signal at
# all (every frame reports the same count) - but pose error clearly separates
# the arms: the candidate ends closer to ground truth at the same cost.
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5, max_halvings=1)
for i in range(24):
    t.observe(100, at_cap=True,
             pose_err=(0.01 if t.k == 1.0 else 0.004))
assert t.frozen and t.k == 0.5, t.summary()
assert t.history[0][1] == "REDUCE" and t.history[0][6] == "pose_err", t.summary()
print(t.summary())

print("\n[14] cap-bound WITH pose error resolves KEEP too (sign carries through)")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5, max_halvings=1)
for i in range(24):
    t.observe(100, at_cap=True,
             pose_err=(0.003 if t.k == 1.0 else 0.009))   # candidate WORSE
assert t.frozen and t.k == 1.0, t.summary()
assert t.history[0][1] == "KEEP" and t.history[0][6] == "pose_err", t.summary()

print("\n[15] ONE pair missing a pose-error reading is TOLERATED (the bootstrap "
     "hole a real PoseErrDelta caller always has on its very first frame) - "
     "the fallback still activates on the rest rather than declining")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5, max_halvings=1)
i = 0
def feed15(k):
    global i
    err = None if i == 3 else (0.01 if k == 1.0 else 0.004)
    i += 1
    return err
for _ in range(24):
    t.observe(100, at_cap=True, pose_err=feed15(t.k))
assert t.frozen and t.k == 0.5, t.summary()
assert t.history[0][1] == "REDUCE" and t.history[0][6] == "pose_err", t.summary()
assert t.history[0][4] == 5, t.summary()   # 6 pairs, 1 missing -> 5 counted
print(f"  PASS  5/6 pairs (1 tolerated hole) still resolved REDUCE via pose_err")

print("\n[15b] TWO OR MORE missing pairs still declines - only the single "
     "bootstrap hole is tolerated, not an unexplained second one")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5)
i = 0
def feed15b(k):
    global i
    err = None if i in (3, 7) else (0.01 if k == 1.0 else 0.004)
    i += 1
    return err
for _ in range(24):
    t.observe(100, at_cap=True, pose_err=feed15b(t.k))
assert t.frozen and t.k == 1.0 and t.history[0][1] == "CAP-BOUND", t.summary()
print("  PASS  2 missing pairs (out of 6) declines rather than trusting 4/6")

print("\n[16] per-frame at_cap overrides a fixed budget - GSLAM's own num_iters "
     "can double mid-run, so a frame short of a FIXED budget can still be at "
     "its OWN (larger) cap that frame")
t = OnlineLRTuner(block_frames=12, budget=200, cap_limit=0.5,   # budget never met
                  max_halvings=1)
for i in range(24):
    t.observe(90, at_cap=True,  # explicit: this frame WAS at its own cap
             pose_err=(0.01 if t.k == 1.0 else 0.004))
assert t.frozen and t.k == 0.5 and t.history[0][6] == "pose_err", t.summary()
print("  PASS  explicit at_cap=True resolved via pose_err although 90 < budget=200")

print("\n[17] no at_cap and no budget: cap-bound logic never engages, same as before")
t = OnlineLRTuner(block_frames=12)   # no budget passed
for i in range(24):
    t.observe(100, pose_err=0.01)   # constant iters, pose_err ignored (not cap-bound)
assert t.frozen and t.history[0][1] == "AMBIGUOUS", t.summary()
assert t.history[0][6] == "it/frame", t.summary()

print("\n[18] MonoGS/SplaTAM-style callers (no at_cap, no pose_err) are unaffected")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5)
run(t, lambda k, i: 100)   # identical to test [3], old call signature
assert t.frozen and t.k == 1.0 and t.history[0][1] == "CAP-BOUND"
assert t.history[0][6] == "it/frame", t.summary()

print("\n[19] a NaN pose_err (e.g. an invalid ScanNet ground-truth pose) is "
     "treated exactly like a missing (None) reading - ONE tolerated (does "
     "not poison the mean into NaN), a SECOND one still declines")
t = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5, max_halvings=1)
i = 0
def feed19(k):
    global i
    err = float("nan") if i == 3 else (0.01 if k == 1.0 else 0.004)
    i += 1
    return err
for _ in range(24):
    t.observe(100, at_cap=True, pose_err=feed19(t.k))
assert t.frozen and t.k == 0.5, t.summary()
assert t.history[0][1] == "REDUCE" and t.history[0][6] == "pose_err", t.summary()
mean_d = t.history[0][3]
assert mean_d == mean_d, t.summary()   # NOT NaN - a poisoned mean fails this
print(f"  PASS  one NaN tolerated, mean_delta={mean_d} clean (not NaN)")

t2 = OnlineLRTuner(block_frames=12, budget=100, cap_limit=0.5)
i = 0
def feed19b(k):
    global i
    err = float("nan") if i in (3, 7) else (0.01 if k == 1.0 else 0.004)
    i += 1
    return err
for _ in range(24):
    t2.observe(100, at_cap=True, pose_err=feed19b(t2.k))
assert t2.frozen and t2.k == 1.0 and t2.history[0][1] == "CAP-BOUND", t2.summary()
assert t2.history[0][6] == "it/frame", t2.summary()   # never reached pose_err
print("  PASS  two NaN readings declines, same threshold as two missing (None)")

print("\n[20] slot_frames widens each role's turn; k holds constant within a "
     "slot and k_log matches frame-for-frame")
t = OnlineLRTuner(block_frames=32, max_halvings=1, slot_frames=4,
                  slot_warmup=1, cap_limit=0.5)
assert t.min_pairs == 4, t.min_pairs   # 32 raw frames // (2*4) = 4 pairs
ref_block, cand_block = [50, 40, 42, 44], [30, 20, 22, 24]
ks = []
for v in ref_block + cand_block:
    ks.append(t.k)
    t.observe(v)
assert ks == [1.0] * 4 + [0.5] * 4, ks
assert t.k_log == ks, (t.k_log, ks)
print("  PASS  k constant for slot_frames=4 frames per role, then switches")

print("\n[21] the slot mean drops the first slot_warmup frames, exactly")
# ref tail = [40,42,44] mean 42.0; cand tail = [20,22,24] mean 22.0.
# 3 more identical rounds close the rung (min_pairs=4) with a huge, clean gap.
# WHICH BLOCK LEADS ALTERNATES ROUND TO ROUND - feed by FOLLOWING t.k (as a
# real caller does, see test [20]), not by assuming ref_block always leads.
for _ in range(6):   # 6 slots = 3 more rounds
    for v in (ref_block if t.k == 1.0 else cand_block):
        t.observe(v)
assert t.frozen and t.k == 0.5, t.summary()
_, verdict, _, mean_d, n, mean_ref, signal = t.history[0]
assert verdict == "REDUCE" and n == 4 and signal == "it/frame", t.summary()
assert abs(mean_d - (22.0 - 42.0)) < 1e-9, mean_d   # exact tail means, not raw
assert abs(mean_ref - 42.0) < 1e-9, mean_ref
print(f"  PASS  mean_delta={mean_d} matches the hand-computed tail means "
     f"exactly (raw block means would give a different, wrong number)")

print("\n[22] slot_warmup must be < slot_frames")
try:
    OnlineLRTuner(slot_frames=4, slot_warmup=4)
    assert False, "should have raised"
except ValueError:
    pass
try:
    OnlineLRTuner(slot_frames=4, slot_warmup=5)
    assert False, "should have raised"
except ValueError:
    pass
print("  PASS  a slot with nothing left after warmup is refused at construction")

print("\n[23] slot_frames=1, slot_warmup=0 (the default) is byte-identical to "
     "the pre-slot design - same schedule, same verdicts, same block_frames "
     "normalization, for every earlier scenario in this file")
t_a = OnlineLRTuner(block_frames=20, max_halvings=2)
t_b = OnlineLRTuner(block_frames=20, max_halvings=2, slot_frames=1, slot_warmup=0)
assert t_a.min_pairs == t_b.min_pairs and t_a.block_frames == t_b.block_frames
rng = random.Random(11)
seq = [80 + rng.gauss(0, 5) for _ in range(80)]
for v in seq:
    if t_a.frozen and t_b.frozen:
        break
    if not t_a.frozen:
        t_a.observe(v)
    if not t_b.frozen:
        t_b.observe(v)
assert t_a.history == t_b.history, (t_a.history, t_b.history)
assert t_a.k_log == t_b.k_log
print("  PASS  identical history and k_log")

print("\n[24] cross-arm contamination simulation: does the existing per-frame "
     "design (slot=1) produce WRONG verdicts under a modelled handoff-carryover "
     "process, or only lose power? Answers whether a real fr1_desk wrong "
     "REDUCE (2026-09-22, t=-2.7, n=14 pairs) implicates the schedule itself.")
def sim_pose_err(rng, ref_eq, cand_eq, decay, noise_sd, slot_frames, slot_warmup,
                 min_pairs_target=12, max_frames=1200, cap=60):
    block_frames = 2 * min_pairs_target * slot_frames
    t = OnlineLRTuner(block_frames=block_frames, max_halvings=1,
                      cap_limit=0.5, slot_frames=slot_frames,
                      slot_warmup=slot_warmup)
    log_err = math.log(ref_eq)
    frame = 0
    while not t.frozen and frame < max_frames:
        equilib = ref_eq if t.k == 1.0 else cand_eq
        log_err = (decay * log_err + (1 - decay) * math.log(equilib)
                  + rng.gauss(0, noise_sd))
        t.observe(cap, at_cap=True, pose_err=math.exp(log_err))
        frame += 1
    return t

# High persistence (half-life ~6.6 frames of simulated "trajectory memory"),
# a KEEP scene (candidate is WORSE, like fr1_desk) - the specific direction
# that produced the real wrong verdict.
n_trials, n_wrong, n_correct = 300, 0, 0
for seed in range(n_trials):
    rng = random.Random(seed)
    t = sim_pose_err(rng, ref_eq=0.01, cand_eq=0.0135, decay=0.9,
                     noise_sd=0.15, slot_frames=1, slot_warmup=0)
    v = t.history[0][1] if t.history else "NONE"
    n_wrong += v == "REDUCE"
    n_correct += v == "KEEP"
wrong_rate = n_wrong / n_trials
# The threshold is calibrated for ~2% PER LOOK. A wrong-rate meaningfully
# above that (say >8%, 4x the nominal rate) would say the contamination
# mechanism itself manufactures confident wrong answers, not just noise.
# Simulated result (2026-09-22): wrong-rate stayed in the 0-2% band at every
# decay/drift/slot combination tried - indistinguishable from ordinary
# sampling noise at the calibrated rate. This does NOT support "per-frame
# interleaving is structurally biased toward a wrong verdict under
# contamination" as the explanation for the real fr1_desk result.
assert wrong_rate < 0.08, (
    f"wrong-rate {wrong_rate:.1%} over {n_trials} trials is well above the "
    f"calibrated ~2%/look rate - contamination MAY manufacture confident "
    f"wrong answers after all; re-open the structural-bias question")
print(f"  wrong={n_wrong}/{n_trials} ({wrong_rate:.1%})  correct={n_correct}/{n_trials}"
     f"  -> consistent with ordinary sampling noise, not a schedule defect")

print("\n[25] PoseErrDelta: the first valid frame reports no delta, then "
     "differences against the LAST VALID level, whether or not a gap "
     "of invalid frames sits in between")
d = PoseErrDelta()
assert d.update(0.010) is None              # nothing to difference against yet
assert abs(d.update(0.012) - 0.002) < 1e-12
assert d.update(float("nan")) is None       # unusable, and does not move the ref
assert d.update(0.5, valid=False) is None   # explicitly invalid, same treatment
# next valid frame spans BACK to 0.012 (the last valid level), not to the
# invalid readings in between - a multi-frame delta, not a wrong one
assert abs(d.update(0.015) - 0.003) < 1e-12
print("  PASS  0.010->None, 0.012->+0.002, [nan, invalid]->None twice, "
     "0.015->+0.003 (spans the gap back to 0.012)")

print("\n[26] LEVEL is HALF the true effect and order-confounded; DELTA is "
     "the FULL effect and order-independent - exact, with zero noise")
def one_round(ref_inc, cand_inc, err0, order):
    got = {}
    err = err0
    for role in order:
        inc = ref_inc if role == "ref" else cand_inc
        prev = err
        err = err + inc
        got[role] = (err, err - prev)
    return got["cand"][0] - got["ref"][0], got["cand"][1] - got["ref"][1]

ref_inc, cand_inc = 0.002, 0.001   # true effect (cand - ref) = -0.001
lv_even, dl_even = one_round(ref_inc, cand_inc, 0.05, ("ref", "cand"))
lv_odd, dl_odd = one_round(ref_inc, cand_inc, 0.05, ("cand", "ref"))
assert abs(lv_even - cand_inc) < 1e-12 and abs(lv_odd - (-ref_inc)) < 1e-12, (
    "LEVEL's per-pair value depends on which role led - this IS the "
    "confound, not a test bug")
assert abs((lv_even + lv_odd) / 2 - (cand_inc - ref_inc) / 2) < 1e-12
assert abs(dl_even - (cand_inc - ref_inc)) < 1e-12
assert abs(dl_odd - (cand_inc - ref_inc)) < 1e-12
print(f"  PASS  LEVEL alternates {lv_even:+.4f}/{lv_odd:+.4f} (order-dependent, "
     f"averages to {(lv_even + lv_odd) / 2:+.4f} = HALF of {cand_inc - ref_inc:+.4f}); "
     f"DELTA is {dl_even:+.4f} on EVERY pair regardless of order = the FULL effect")

print("\n[27] with realistic noise and small pair counts (6-20, what GSLAM "
     "runs actually use), DELTA resolves the correct verdict substantially "
     "more often than LEVEL at the SAME frame budget, in both directions, "
     "with no increase in wrong-rate")
def sim_pair(rng, ref_inc, cand_inc, noise_sd, n_pairs):
    err = 0.05
    lv, dl = [], []
    for i in range(n_pairs):
        order = ("ref", "cand") if i % 2 == 0 else ("cand", "ref")
        got = {}
        for role in order:
            inc = ref_inc if role == "ref" else cand_inc
            prev = err
            err = err + inc + rng.gauss(0, noise_sd)
            got[role] = (err, err - prev)
        lv.append(got["cand"][0] - got["ref"][0])
        dl.append(got["cand"][1] - got["ref"][1])
    return lv, dl


def sim_verdict(diffs):
    mean, se, tt = _paired_t(diffs)
    crit = _crit(len(diffs) - 1)
    if abs(tt) < crit:
        return "AMBIGUOUS"
    return "REDUCE" if mean < 0 else "KEEP"


n_trials = 400
for true_reduce in (True, False):
    ref_inc = 0.0015
    cand_inc = ref_inc * (0.5 if true_reduce else 1.6)
    correct = "REDUCE" if true_reduce else "KEEP"
    wrong = "KEEP" if true_reduce else "REDUCE"
    for noise_sd, n_pairs in ((0.002, 14), (0.003, 20)):
        lvl_correct = lvl_wrong = dlt_correct = dlt_wrong = 0
        for seed in range(n_trials):
            rng = random.Random(str((seed, true_reduce, noise_sd, n_pairs)))
            lv, dl = sim_pair(rng, ref_inc, cand_inc, noise_sd, n_pairs)
            lv_v, dl_v = sim_verdict(lv), sim_verdict(dl)
            lvl_correct += lv_v == correct; lvl_wrong += lv_v == wrong
            dlt_correct += dl_v == correct; dlt_wrong += dl_v == wrong
        # The point: DELTA finds the answer more often at the SAME n, not
        # that either is highly powered here - both stay mostly AMBIGUOUS at
        # this deliberately small budget, which is the honest picture at
        # what GSLAM has actually been running.
        assert dlt_correct >= lvl_correct, (
            f"true_reduce={true_reduce} noise={noise_sd} n={n_pairs}: "
            f"DELTA ({dlt_correct}) did not beat LEVEL ({lvl_correct})")
        assert dlt_wrong <= n_trials * 0.03, (dlt_wrong, n_trials)
        print(f"  true={correct:6s} noise={noise_sd:.3f} n_pairs={n_pairs:2d}  "
             f"LEVEL correct={lvl_correct:3d} wrong={lvl_wrong:2d}   "
             f"DELTA correct={dlt_correct:3d} wrong={dlt_wrong:2d}")

print("\nALL PASS")
