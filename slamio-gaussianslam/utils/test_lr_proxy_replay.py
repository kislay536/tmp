import contextlib
import io
import math
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "profiling", "lr_proxy_replay.py")


def make_log(frames):
    """frames: list of (k, iters, rel_best, rel_final) in frame order."""
    fmt = lambda xs: ",".join(str(x) for x in xs)
    n = len(frames)
    return (
        "noise line\n"
        f"  IT/FRAME SERIES n=2: 1,1\n"                      # an earlier, shorter print
        f"  IT/FRAME SERIES n={n}: {fmt(f[1] for f in frames)}\n"
        f"  REL-GRAD BEST SERIES n={n}: {fmt(f[2] for f in frames)}\n"
        f"  REL-GRAD FINAL SERIES n={n}: {fmt(f[3] for f in frames)}\n"
        f"  K SERIES n={n}: {fmt(f[0] for f in frames)}\n"
        "51%|##| 3/8 [00:01]\r 52%|##| 4/8[OnlineLRTuner] frozen\n")


def schedule(n_pairs, ref=1.0, cand=0.5):
    """Frame order of the tuner: pair i even -> (ref, cand), odd -> (cand, ref),
    then a few frames at one k after the freeze."""
    ks = []
    for i in range(n_pairs):
        ks += [ref, cand] if i % 2 == 0 else [cand, ref]
    return ks + [cand] * 4


def run(text):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-B", SCRIPT, path],
                           capture_output=True, text=True)
    finally:
        os.unlink(path)
    return r.returncode, r.stdout


print("[0] uncapped, real effect on BOTH signals -> both read REDUCE-like")
rng = random.Random(1)
frames = []
for k in schedule(14):
    base = 80 + rng.gauss(0, 6)
    frames.append((k, round(base * (0.8 if k == 0.5 else 1.0)),
                   round(0.05 * (0.6 if k == 0.5 else 1.0)
                         * math.exp(rng.gauss(0, 0.15)), 5),
                   round(0.08 * (0.6 if k == 0.5 else 1.0)
                         * math.exp(rng.gauss(0, 0.15)), 5)))
rc, out = run(make_log(frames))
print(out)
assert rc == 0, out
assert out.count("REDUCE-like") == 3, out
assert "14 pairs" in out and "CONFLICT" not in out, out

print("[1] CAPPED: it/frame is a constant, the stand-in still resolves")
rng = random.Random(2)
frames = [(k, 40, round(0.2 * (0.5 if k == 0.5 else 1.0)
                        * math.exp(rng.gauss(0, 0.15)), 5),
           round(0.3 * (0.5 if k == 0.5 else 1.0)
                 * math.exp(rng.gauss(0, 0.15)), 5))
          for k in schedule(14)]
rc, out = run(make_log(frames))
print(out)
assert rc == 0
assert "unresolved" in out.split("rel-grad best")[0], out       # it/frame: t = 0
assert "REDUCE-like" in out.split("rel-grad best")[1].splitlines()[0], out
assert "100% of frames at cap" in out, out

print("[2] the signals point OPPOSITE ways -> flagged")
rng = random.Random(3)
frames = [(k, round((80 + rng.gauss(0, 4)) * (0.8 if k == 0.5 else 1.0)),
           round(0.05 * (2.0 if k == 0.5 else 1.0)
                 * math.exp(rng.gauss(0, 0.1)), 5), 0.1)
          for k in schedule(14)]
rc, out = run(make_log(frames))
print(out)
assert "CONFLICT" in out, out

print("[3] a log with no new series fails cleanly, not with a traceback")
rc, out = run("IT/FRAME SERIES n=3: 1,2,3\n")
assert rc == 2 and "predates the logging" in out, (rc, out)

print("[4] frames after the freeze (same k twice) and NaN frames are skipped")
frames = [(1.0, 60, 0.1, 0.1), (0.5, 50, float("nan"), 0.1),
          (0.5, 50, 0.05, 0.05), (0.5, 50, 0.05, 0.05)]
text = make_log(frames).replace("nan", "nan")
rc, out = run(text)
assert rc == 0 and "1 pairs" in out, out

print("\nALL PASS")
