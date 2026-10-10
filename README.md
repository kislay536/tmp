# slamio runs (SplaTAM · Gaussian-SLAM · MonoGS) — setup and run guide

This branch contains everything needed to run the **24 slamio benchmark runs** on an NVIDIA H100 and collect the results in one CSV.

**What slamio is:** an optimised tracking/mapping stack (pose preconditioner, convergence-based early stopping, CUDA-graph capture of a tracking iteration, stale-gradient reuse, fixed-capacity binning, adaptive mapping budget, frame prefetch) that is built directly into forks of three 3DGS-SLAM systems. It shares one modified CUDA rasterizer (`diff-gaussian-rasterization`). Source: the `3DGS-framework` repo, branch `slamio_final`, commit `475bf6c` (see `slamio-*/SOURCE_COMMIT`), plus the small harness changes listed at the end.

## The 24 runs

For every model, scene and mode there is one run. "baseline" is the **same slamio code with every slamio feature switched off** (Adam pose optimiser, fixed iteration budget, no graph/reuse/early-stop/adaptive mapping), so baseline and slamio differ only in the optimisations. "optimization" is slamio-final.

| Model | Scenes | Modes | Threads | Runs |
|---|---|---|---|---|
| SplaTAM | fr1_desk (TUM), room0 (Replica), scene0059_00 (ScanNet) | baseline, optimization | – | 6 |
| Gaussian-SLAM | same 3 | baseline, optimization | – | 6 |
| MonoGS | same 3 | baseline, optimization | single, multi | 12 |

CSV label (`optimization_type` column): `slamio_baseline`, `slamio_optimization`; MonoGS adds `_singlethread` / `_multithread`.

## Layout

```
slamio-splatam/        SplaTAM/        fork (scripts/splatam.py, configs/)
slamio-gaussianslam/   Gaussian-SLAM/  fork (run_slam.py, configs/)
slamio-monogs/         MonoGS/         fork (slam.py, configs/)
   each also has: utils/ (slamio modules), submodules/ (CUDA extensions), cells/ (captured slamio-final settings)
launch.sh              runs one combo, or all 24 with --all --only-opt slamio; appends a row to the results CSV
resume_slamio.sh       runs the 24 combos, SKIPPING any already in the CSV (use this; it is safe to restart)
watch_slamio.sh        live progress table
extract_*_metrics.py   turn a finished run into the CSV fields
setup_env.sh           builds the three venvs + CUDA extensions
setup_data.sh          links the datasets where each fork expects them
scripts/download_*.sh  dataset download helpers
reference/             slamio rows measured on a Jetson Thor (sanity check only, not comparable timings)
```

## 1. Requirements

- Linux x86_64 with an H100 (other GPUs work: set `TORCH_CUDA_ARCH_LIST`, e.g. 8.0 for A100)
- NVIDIA driver, **CUDA toolkit 12.x with `nvcc`** (12.1 matches the torch wheels; 12.x up to 12.8 works for compiling)
- Python **3.10**, `git`, `g++`, `wget`, `unzip`, ~150 GB free disk (venvs ≈ 3×3 GB, datasets, run outputs)
- Only one run may use the GPU at a time. Do not start two sweeps.

## 2. Build the environments

```bash
git clone -b slamio-h100 https://github.com/kislay536/tmp.git slamio && cd slamio
# optional overrides: PYTHON=/path/python3.10 CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=9.0
bash setup_env.sh
```
Creates `slamio-*/<Fork>/.venv` with torch 2.3.1+cu121 (the version slamio was validated on), compiles the rasterizer (and `simple_knn` for MonoGS / Gaussian-SLAM) for the local GPU, and runs a smoke test per environment. Every environment must end with `=== … ready ===`. The smoke test also reports whether Open3D has CUDA — Gaussian-SLAM on Replica room0 needs it (the Linux x86_64 pip wheel normally does).

## 3. Get the data

```bash
export DATA_ROOT=$PWD/data            # any location with ~60 GB free
bash scripts/download_tum.sh fr1_desk
bash scripts/download_replica.sh room0
bash scripts/download_scannet.sh scene0059_00   # needs ScanNet access (Terms of Use) – see the script header
bash setup_data.sh
```
Expected layout under `$DATA_ROOT`: `TUM_RGBD/rgbd_dataset_freiburg1_desk/`, `Replica/room0/` (`results/`, `traj.txt`), `ScanNet/scene0059_00/` (`color/ depth/ pose/ intrinsic/`). `setup_data.sh` only creates symlinks and stops with a clear message if a scene is missing.

## 4. Smoke test (≈10 min, recommended)

```bash
SLAMIO_EXTRA_ENV="MONOGS_FRAMES=40" bash launch.sh --model monogs --optimization slamio \
    --scene fr1_desk --mode optimization --thread multi --run-tag smoke --csv /tmp/smoke.csv
cat /tmp/smoke.csv        # one row, no empty metric columns
```
`SLAMIO_EXTRA_ENV` appends `VAR=value` pairs to the run's environment; use it only for tests (frame caps: `FRAMES=` Gaussian-SLAM and SplaTAM baseline, `MONOGS_FRAMES=` MonoGS, `SPLATAM_FINAL_FRAMES=` SplaTAM slamio-final). Capped runs skip some evaluations, so Gaussian-SLAM may print "metric extraction failed" for a short run — that is expected.

## 5. Run all 24

```bash
mkdir -p logs && nohup setsid bash resume_slamio.sh > logs/resume_slamio_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
TOTAL_COMBOS=24 bash watch_slamio.sh          # live view (Ctrl+C to leave; --once for a snapshot)
```
- Combos run one after another in a fixed order. A row is appended to the CSV only when a run finishes.
- **If the machine reboots or the job dies, start the same command again**: finished combos are skipped, the interrupted one restarts from scratch.
- A failed combo does not stop the sweep; it is listed at the end. Per-run logs: `logs/slamio-<model>/<scene>_<mode>[_<n>thread].log`; run outputs: `logs/slamio-<model>/runs/…`.
- Rough time on an H100: the earlier non-slamio baseline runs on this hardware summed to ≈ 12 h for one pass over these 12 model/scene cells; with the two modes and MonoGS in both thread modes expect **about 0.5–1 day** in total. SplaTAM on Replica room0 is the longest single run.

## 6. Results

The CSV is `experiments/results.csv` on an H100 (`experiments/results_<gpu>.csv` on other GPUs; override with `CSV=/path resume_slamio.sh`). 15 columns:

`model, dataset, scene, optimization_type, gpu, compute_location, total_time_s, tracking_ms_per_frame, tracking_ms_per_iter, mapping_ms_per_frame, mapping_ms_per_iter, ate_cm, psnr_db, tracking_iters_per_frame, mapping_iters_per_frame`

- `total_time_s` = tracking + mapping wall time (evaluation, model loading and, for Gaussian-SLAM, visual-odometry time are **excluded**).
- Compare `slamio_baseline` with `slamio_optimization` for the same model/scene (and thread mode). The speed-up should come with a similar `ate_cm` and `psnr_db`.
- MonoGS **multi-thread** leaves the mapping columns empty (the backend runs in a separate process); single-thread fills them. Mapping is reported per `map()` call.
- `ate_cm` is the aligned ATE RMSE in cm.
- `reference/thor_results_slamio.csv` holds the rows finished on a Jetson Thor before handoff. Use them only to check that accuracy (ATE/PSNR) is in the same range; timings are not comparable across GPUs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `setup_env.sh`: nvcc not found / wrong Python | set `CUDA_HOME` / `PYTHON` (must be 3.10) |
| `no kernel image is available` | rebuild with the right `TORCH_CUDA_ARCH_LIST` (`bash setup_env.sh`) |
| Gaussian-SLAM room0 fails at the first tracked frame with an Open3D CUDA error | the venv's Open3D has no CUDA: `pip install -U open3d` in `slamio-gaussianslam/Gaussian-SLAM/.venv` |
| Gaussian-SLAM faiss errors | `FAISS_GPU=0 bash resume_slamio.sh …` uses the pure-torch fallback (the Thor runs used this) |
| Gaussian-SLAM ScanNet: `IndexError: list index out of range` | `slamio-gaussianslam/datasets/ScanNet` link missing — rerun `setup_data.sh` |
| System runs out of memory / hangs during a very long run | watch `free -g`; restart with the same command, it resumes at the interrupted combo |
| `ERROR: another sweep is already running` | a previous `resume_slamio.sh` / `launch.sh --all` is still alive (`pgrep -af launch.sh`) |

## Known caveats

- **Replica room0** slamio-final settings come from the framework's accepted Replica cells (`slamio-*/cells/room0_optimization.*`); the framework's production cells are only TUM fr1_desk and ScanNet scene0059_00. Treat room0 slamio numbers as untuned.
- The SplaTAM ScanNet slamio-final preset was only checked up to mapping step ≈ 750 on the Thor before handoff; run it to completion first if you want to find problems early (`bash launch.sh --model splatam --optimization slamio --scene scene0059_00 --mode optimization`).
- MonoGS "single-thread" uses the `single_thread` config flag (same convention as the other MonoGS repos in the comparison), not slamio's `INLINE` mode.
- On the Thor, Gaussian-SLAM used a pure-torch neighbour search (`FAISS_GPU=0`) instead of faiss; this branch defaults to faiss-gpu (`FAISS_GPU=1`). Expect small mapping-time differences.
- Harness changes relative to the framework's own scripts: `fps_metrics.json` and odometer timing in `Gaussian-SLAM/src/entities/gaussian_slam.py`; `Eval:` timing lines in `MonoGS/utils/slam_{frontend,backend}.py`; `*_singlethread_on.yaml` MonoGS configs; `SplaTAM/configs/replica/splatam_baseline_room0.py`; an optional `FRAMES` env var in the stock SplaTAM configs (default unchanged: full sequence).
- Torch 2.3.1 is the validated stack. The Thor used torch 2.11 (CUDA 13), which also ran, but CUDA-graph capture is the first thing to check if you move to a newer torch.
