import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["freiburg1_desk", "freiburg1_desk2", "freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household"]

seed = int(0)
scene_name = scenes[int(0)]

map_every = 1
keyframe_every = 5
mapping_window_size = 20
tracking_iters = 200
mapping_iters = 30
scene_radius_depth_ratio = 2

group_name = "TUM"
run_name = f"{scene_name}_seed{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=500, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # ["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=False,
    wandb=dict(
        entity="theairlab",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data/TUM_RGBD",
        gradslam_data_cfg=f"./configs/data/TUM/{scene_name}.yaml",
        sequence=f"rgbd_dataset_{scene_name}",
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=int(os.environ.get("FRAMES", "-1")),
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.002,
        ),
        cuda_graph=dict(
            # Captures only optimizer.step()+zero_grad() (cam_unnorm_rots/
            # cam_trans, the sole two params left in the tracking Adam) into
            # a CUDA graph, replayed on every iteration after a short eager
            # warmup. The rasterizer's own forward/backward calls can't be
            # captured without patching the shared CUDA submodule (a
            # synchronous cudaMemcpy inside rasterizer_impl.cu's forward is
            # illegal during graph capture), so this only ever affects a
            # small slice of each iteration's kernel launches - disabled
            # until a real A/B (run_benchmarks.sh + check_launch_gaps.py)
            # confirms it's worth turning on by default.
            # Verified correct via profiling/verify_tracking_cuda_graph.py
            # (pose/loss trajectory diffs within eager-vs-eager noise floor).
            # --no-profile A/B with this on: 258s wall (vs nsys-profiled
            # 339s/431s figures, which include profiler-instrumentation
            # overhead that scales with kernel-launch count and inflates
            # the apparent win - see the --no-profile baseline run for the
            # real, uninstrumented comparison before trusting a headline %).
            enabled=False,
            warmup_iters=3,
        ),
        pixel_sample=dict(
            # Matches exp/sparse_and_early_stop's v2 tuned SplaTAM config
            # (commit 7c51595): always-sparse [0.0, 1.0] window,
            # sample_ratio=0.6. That combined run (with its own, more
            # aggressive early_stop: min_iters=10) measured 2.16x speedup,
            # +3.3% ATE, +0.58dB PSNR on TUM fr1_desk - a real GPU-time
            # saving (masked tiles skip the CUDA forward/backward kernel
            # work entirely, not just Python-side loss masking), unlike
            # coarse_to_fine which broke on this branch or adaptive_pruning
            # which measured worse on both speed and ATE. Untested
            # combined with this branch's current early_stop tuning
            # (min_iters=70, different from the validated run's 10).
            # Enabled for testing now that PixelSampleTracker's
            # construction/summary prints exist to confirm engagement -
            # the earlier ATE 6.70cm run turned out to be a false alarm
            # (log showed "disabled", never actually ran).
            #
            # Currently OFF: parked while the early_stop retune settings and
            # the on-GPU scratch-tensor change (ca4d889) are measured, so
            # tile masking isn't moving underneath those numbers. Re-enable
            # and A/B on its own once those are settled.
            enabled=False,
            sample_ratio=0.6,
            full_start_ratio=0.0,
            full_end_ratio=1.0,
            gradient_frac=0.5,
        ),
        early_stop=dict(
            # min_iters/patience/retune structure match exp/early-stopping-
            # tracking's / exp/dynamic_early_stopping's validated config
            # (TUM fr1_desk, 1.90x speedup, -1.2% ATE), with min_iters kept
            # at 70 rather than reverted to those branches' 10 (a real run
            # on this branch showed 70 clearly improving ATE over 50).
            #
            # loss_eps/pose_eps restored to the 1e-4/1e-4 reference pair
            # (the values at 7faeedd, before c752962/30129cc replaced them
            # with 8e-4/0.0 copied out of a real retune's output).
            #
            # The argument for the copied pair was that 1e-4/1e-4 can't fire
            # pre-retune - a run's pose diag showed tail_mass=1.00 against a
            # 0.60 threshold and a mean pose delta of 6.2e-4, well above a
            # 1e-4 threshold, so the AND criterion rarely completes. That is
            # accurate but it is a description of early stopping being
            # *conservative*, not broken: seeding the startup thresholds
            # from a retune's own output pre-loosens the criterion for every
            # frame before the first retune, and the retunes themselves only
            # ever loosen further (elbow detection is degenerate on this
            # scene - a real run fit a median elbow of 3 against a floor of
            # 70 - so min_iters never moves and loss_eps is the only knob
            # retuning actually turns).
            #
            # Consequence to expect: frames before the first retune run
            # closer to the full iteration budget. That is the intended
            # trade - the looser startup values correlate with the ATE
            # regressions the one-shot retune experiments kept hitting.
            enabled=True,
            min_iters=70,
            loss_eps=1e-4,
            patience=3,
            pose_eps=1e-4,
            log_signals="../results/signals/splatam_tum_fr1_signals.jsonl",
            # Back to periodic retuning (the pre-6adc503 setting, restored
            # from 9d188b5) after the one-shot variants underperformed.
            # No retune_start key on purpose: it defaults to retune_every,
            # so _frames_since_retune starts at 0 and the first probe opens
            # once 100 resets have happened - i.e. at frame 99 (0-indexed),
            # with frames 0-98 running on the static thresholds above.
            # On the full 573-frame run that gives 4 probes, opening at
            # frames 99 / 213 / 327 / 441 (the counter only advances outside
            # probe mode, so each cycle is 15 probe frames + 100 normal
            # ones). A 5th probe would be due at frame 555 but is suppressed
            # by the end-of-dataset guard, which needs retune_frames + 10 =
            # 25 frames left and only has 17. Each probe is 15 frames capped
            # at 75% of the observed max iters, so the thresholds get
            # re-fitted as the map matures rather than once against the thin
            # early map.
            #
            # retune_frames=15 (up from the 10 this was restored with) buys
            # 50% more signal per elbow fit for ~600 extra tracking
            # iterations across the run - 4 x 15 probe frames instead of
            # 5 x 10, at the same 0.75 cap. Still cheaper than the one-shot
            # variant it replaced (40 probe frames at no cap).
            #
            # On a 100-frame quick run the first probe opens at frame 99 and
            # cannot fill, so quick runs retune 0x and carry no probe
            # overhead at all - which also means they no longer exercise
            # this code path.
            retune_every=100,
            retune_frames=15,
            retune_probe_max_iters_frac=0.75,
            # Kept with min_iters=70 (see exp_dynamic_early_stopping memory:
            # without a matching floor, the very next retune's elbow
            # detection clamps min_iters back down toward the class
            # default floor of 8, undoing the kept value above).
            retune_min_iters_floor=70,
            dataset_frames=573,  # freiburg1_desk
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
        adaptive_pruning=dict(
            # REAL RESULT (TUM fr1_desk, 592 frames): 268,443 Gaussians
            # removed over 149 calls, both axes moved the wrong way -
            # wall time 1663s vs 1468s disabled (slower, likely allocator
            # churn from remove_points() reallocating every parameter
            # tensor + optimizer state on ~each of the last 149 frames),
            # and ATE 4.86cm vs 3.90cm disabled (worse). Opacity+age
            # isn't a safe-enough removal signal - unlike RTGS's original
            # gradient-informed ranking, low opacity alone doesn't mean
            # "not contributing" (thin surfaces / grazing-angle
            # observations can be low-opacity and still useful). Disabled
            # again - this isn't a tuning problem, it's the wrong signal.
            enabled=False,
            start_progress=0.75,
            max_prune_fraction_per_step=0.03,
            min_age_protect=20,
            opacity_threshold=0.05,
            min_opacity_protect=0.85,
        ),
    ),
    adaptive_mapping=dict(
        enabled=True,
        min_iters=22,  # >20 so the prune_every=20/stop_after=20 event always fires
        max_iters=mapping_iters,  # 30 for TUM
        depth_error_ref=0.01,   # TUM real data: noisier, higher error baseline
        color_error_ref=0.05,   # TUM real data: noisier, higher error baseline
        # ~200 new Gaussians/frame = full budget (smaller images). Used to be
        # dead for SplaTAM specifically - splatam.py fed the same already-
        # normalised unseen_ratio into both signals, so this reference was
        # never divided into anything. splatam.py now passes the true raw
        # new-points count alongside a ratio computed from THIS value, same
        # contract AdaptiveMapper.compute_budget() already uses for
        # Gaussian-SLAM, so it actually matters now - and, separately, can be
        # self-calibrated the same way GSLAM's ScanNet cell is
        # (calibrate_new_pts_frames > 0, see splatam_precond.py's ADMAP=1
        # block; NOT the same knob as calibration_keyframes, which only fits
        # depth_error_ref/color_error_ref).
        n_new_pts_ref=200,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    ),
)