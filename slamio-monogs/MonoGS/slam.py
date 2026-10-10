import os
import queue as thread_queue
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb

# Put the repo root on the path BEFORE the `utils.*` imports below. The
# top-level utils/ has no __init__.py, so `utils` is a PEP 420 namespace
# package spanning repo-root/utils and MonoGS/utils - which is how
# utils.binning_capacity and utils.camera_utils both resolve. slam_frontend.py
# does the same insert, but it happens too late to help imports in this file.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gaussian_splatting.scene.gaussian_model import GaussianModel  # noqa: E402
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue, InlineBackendQueue
from utils.prefetch import FramePrefetcher
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        # Frame prefetch (utils/prefetch.py, shared with SplaTAM and
        # Gaussian-SLAM). Default OFF.
        #
        # The dataset is forced onto the CPU when this is enabled, because its
        # __getitem__ then runs on a worker thread and the worker must stay
        # CUDA-free. Camera.init_from_dataset does the H2D on the main thread
        # instead. Nothing else changes: with prefetch off the dataset keeps
        # its own device and every path below is untouched.
        _pf_cfg = config.get("Training", {}).get("prefetch", {})
        if _pf_cfg.get("enabled", False):
            self.dataset.device = "cpu"
            self.dataset = FramePrefetcher(self.dataset, _pf_cfg, name="MonoGS")
            Log(
                f"Frame prefetch ON (queue_depth={self.dataset.queue_depth}); "
                f"dataset moved to CPU, H2D deferred to Camera.init_from_dataset"
            )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # use_inline_backend: test whether eliminating the cross-process
        # CUDA context split (frontend/backend normally run as separate
        # mp.Process, each with its own context) helps, at the cost of the
        # continuous background-mapping loop (never runs here - see
        # InlineBackendQueue) and all tracking/mapping overlap. Ported from
        # RTGS (github.com/UMN-ZhaoLab/RTGS, MonoGS_fullend branch) - see
        # rtgs_concurrency_and_benchmark_caveat memory for why our own
        # single_thread flag does NOT test this (it keeps backend as a
        # separate mp.Process regardless, only changing sync behavior).
        self.use_inline_backend = self.config["Training"].get(
            "use_inline_backend", False
        )
        # INLINE=0/1 overrides it without editing the YAML.
        #
        # WHY THIS MATTERS FOR EVERY TIMING COMPARISON HERE. Inline runs the
        # backend SYNCHRONOUSLY in the frontend's CUDA context - no
        # tracking/mapping overlap - and under it iter_per_kf jumps to the full
        # mapping_itr_num, so each keyframe's mapping pass is far deeper. Both
        # make wall time longer by construction.
        #
        # The recorded main baseline (726.658s, FPS 0.815) predates this flag
        # and was therefore ASYNC, so comparing any inline run against it
        # measures the concurrency model rather than whatever is being tested.
        # The valid control for an inline arm is the same config with the arm
        # switched off - see the rtgs_concurrency_and_benchmark_caveat note.
        if "INLINE" in os.environ:
            self.use_inline_backend = os.environ["INLINE"] not in (
                "0", "", "false", "False")
        if self.use_inline_backend and self.use_gui:
            raise NotImplementedError(
                "use_inline_backend does not support use_gui - the GUI "
                "process is only started alongside the multiprocessing "
                "backend path. Set Results.use_gui: False to use "
                "use_inline_backend."
            )
        if self.use_inline_backend:
            frontend_queue = thread_queue.Queue()
            backend_queue = InlineBackendQueue()
        else:
            frontend_queue = mp.Queue()
            backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        if self.use_inline_backend:
            backend_queue.set_backend(self.backend)
            # Lets FrontEnd.run() call self.backend.idle_mapping() directly,
            # substituting for BackEnd.run()'s continuous background-
            # mapping loop, which never executes in this mode.
            self.frontend.backend = self.backend
            self.frontend.inline_backend = True

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        if self.use_inline_backend:
            Log("Running frontend with inline backend (single CUDA context)")
            self.frontend.run()
        else:
            backend_process = mp.Process(target=self.backend.run)
            if self.use_gui:
                gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
                gui_process.start()
                time.sleep(5)

            backend_process.start()
            self.frontend.run()
            backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        # Only correct in inline mode: use_inline_backend runs self.backend in
        # THIS process (frontend.backend = self.backend, slam.py:165), so its
        # counters are the real, mutated ones. In the default separate-process
        # mode, self.backend here is the pre-fork copy - its own summary is
        # printed from inside slam_backend.py's run() instead, in the child.
        if self.use_inline_backend:
            Log(self.backend.mapping_summary())

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # COLOUR REFINEMENT IS 26,000 FIXED MAPPING ITERATIONS, and by
            # this line every number a tracking measurement needs is already
            # final: ATE, total time, FPS and iterations/frame have all been
            # logged above. Refinement cannot change a pose. It only improves
            # the rendering metrics reported as "after_opt".
            #
            # WHY IT CAN BE SKIPPED FOR A PROFILE. It runs outside the
            # monogs_tracking NVTX range, so profiling/sparse_stage_split.py
            # discards every kernel it launches - the trace pays for it and
            # the answer never uses it. Under nsys that is 26,000 iterations
            # of render + backward added to a capture already holding
            # ~40,000 tracking steps, and it was observed to stall the
            # capture outright rather than merely slow it.
            #
            # PROFILING ONLY, AND IT CHANGES A REPORTED NUMBER. With this
            # set, the "after_opt" psnr/ssim/lpips are the UNREFINED values
            # and will sit far below this model's published rendering
            # quality. Never quote rendering metrics from a run that set it.
            # ATE and the tracking iteration counts are untouched - both are
            # already final above.
            _skip_refine = os.environ.get("MONOGS_SKIP_REFINE", "0") == "1"
            if _skip_refine:
                Log("SKIPPING color refinement (MONOGS_SKIP_REFINE=1); "
                    "after_opt rendering metrics below are UNREFINED and "
                    "must not be quoted", tag="Eval")
            else:
                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        if not self.use_inline_backend:
            # DRAIN BEFORE JOINING, OR join() NEVER RETURNS.
            #
            # mp.Queue hands items to a background feeder thread, and a process
            # will not exit while that thread still has items pending. The
            # backend pushes "sync_backend" packets carrying GAUSSIANS - CUDA
            # tensors - and anything the frontend did not consume pins the
            # backend open forever. The symptom is a run that prints every
            # summary, including the backend's own last line, and then hangs.
            #
            # This only bites with the iteration graph on: those runs produce
            # ~119 sync_backend messages against ~9 without it, which is why a
            # ten-run graph-off loop completed cleanly and the first graph-on
            # loop stalled on run 1. Everything needed has already been read by
            # here - the eval block above pulls the final Gaussians - so
            # discarding the remainder is safe.
            while True:
                try:
                    if frontend_queue.empty():
                        break
                    frontend_queue.get_nowait()
                except Exception:  # noqa: BLE001 - empty/closed, nothing to do
                    break

            # And do not trust the drain alone. A bounded join plus terminate
            # turns "the experiment stalls overnight" into "one run loses its
            # teardown", which is the right trade for a batch of A/B runs.
            backend_process.join(timeout=60)
            if backend_process.is_alive():
                Log("Backend did not exit within 60s - terminating it")
                backend_process.terminate()
                backend_process.join(timeout=10)

        # Shut the prefetch worker down explicitly rather than relying on it
        # being a daemon. It normally retires on its own once the dataset is
        # exhausted, but a run that stops early leaves it parked on a full
        # queue, and a thread nobody stopped is not something to leave behind
        # in a process that is trying to exit.
        if hasattr(self.dataset, "close"):
            self.dataset.close()

        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    # MONOGS_PROBE=1 strips everything the acquisition-lr probe does not read.
    #
    # That probe runs ~30 frames at two learning rates and compares the
    # per-frame iteration counts; it reads NOTHING else. Meanwhile a normal run
    # follows tracking with eval_rendering, which carries the ATE evaluation,
    # the rendering metrics AND the 26000-iteration colour refinement - none of
    # which a 30-frame diagnostic can produce a meaningful number from anyway,
    # since a truncated run's ATE and PSNR are meaningless by construction.
    #
    # It also turns off save_results, so probe runs leave no output directory
    # to be mistaken for a real one later.
    if os.environ.get("MONOGS_PROBE", "0") == "1":
        Log("MONOGS_PROBE=1: eval_rendering, save_results and the colour "
            "refinement are OFF. Diagnostic run - only the per-frame "
            "iteration series is valid.")
        config["Results"]["eval_rendering"] = False
        config["Results"]["save_results"] = False
        config["Results"]["use_gui"] = False

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
