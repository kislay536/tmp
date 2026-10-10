import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
    getWorld2View2_capture_safe,
)
from utils.slam_utils import image_gradient, image_gradient_mask


# Module-level rather than per-Camera because Cameras are constructed in four
# places (init_from_dataset, init_from_gui, the backend, the GUI) and the flag
# has to hold for all of them. Set once, from the frontend, when the tracking
# iteration graph is enabled.
_CAPTURE_SAFE_POSE = False


def set_capture_safe_pose(enabled: bool):
    """Switch Camera onto the pose path that a CUDA graph can record.

    Two things change, and both are mathematically identities - see
    getWorld2View2_capture_safe and camera_center below. They are still gated,
    because they move floating-point roundoff and therefore move ATE in the
    last digits, and an ungated change would silently invalidate every A/B
    against the graph-off arm.

    OFF (default): byte-for-byte the upstream behaviour.
    ON: no torch.linalg.inv anywhere in the render path, and R/T are updated in
        place so a captured render keeps reading the buffer update_pose writes.
    """
    global _CAPTURE_SAFE_POSE
    _CAPTURE_SAFE_POSE = bool(enabled)
    print(f"[Camera] capture-safe pose path: {_CAPTURE_SAFE_POSE}", flush=True)


def capture_safe_pose() -> bool:
    return _CAPTURE_SAFE_POSE


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, device=None):
        gt_color, gt_depth, gt_pose = dataset[idx]

        # THE HOST-TO-DEVICE COPY HAPPENS HERE, ON THE CALLING THREAD. That
        # placement is load-bearing, not incidental.
        #
        # With Training.prefetch enabled the dataset is constructed on CPU and
        # dataset[idx] above runs on a PREFETCH WORKER THREAD, which must not
        # touch CUDA: graph capture uses cudaStreamCaptureModeGlobal, where any
        # thread launching into a non-capturing stream is an error. See the
        # contract in utils/prefetch.py.
        #
        # With prefetch off the dataset returns CUDA tensors already and both
        # .to() calls are no-ops, so this is behaviour-preserving either way.
        device = device if device is not None else dataset.device
        if torch.is_tensor(gt_color):
            gt_color = gt_color.to(device=device, non_blocking=True)
        if torch.is_tensor(gt_pose):
            gt_pose = gt_pose.to(device=device, non_blocking=True)

        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        if _CAPTURE_SAFE_POSE:
            return getWorld2View2_capture_safe(self.R, self.T).transpose(0, 1)
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        if _CAPTURE_SAFE_POSE:
            # world_view_transform is w2c transposed, so
            #   world_view_transform.inverse()[3, :3]
            #     = ((Rt^T)^-1)[3, :3] = ((Rt^-1)^T)[3, :3] = (Rt^-1)[:3, 3]
            # and for Rt = [[R, t], [0, 1]] the inverse's translation block is
            # exactly -R^T t. Same value, no matrix inversion, no host readback.
            return -(self.R.transpose(0, 1) @ self.T)
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        if _CAPTURE_SAFE_POSE:
            # IN PLACE, and this is the whole reason the capture-safe flag
            # exists.
            #
            # Rebinding self.R/self.T to new tensor objects works fine eagerly,
            # but it silently breaks under a captured iteration. The render at
            # the top of the iteration reads whatever tensors self.R/self.T
            # named AT CAPTURE TIME; update_pose at the bottom of the same
            # iteration then rebinds them to fresh tensors in the graph's
            # private pool. Replay re-executes the recorded kernels, not the
            # Python rebinding - so every replay renders from the pose the
            # capture iteration started with, forever. The loop would run its
            # full iteration count, capture 100% of frames, and optimise
            # nothing.
            #
            # Writing into a fixed buffer makes the read-then-write a genuine
            # data dependency inside the graph, which is what the eager loop
            # actually means. Every reader in the codebase re-reads .R/.T at
            # use time or clones (slam_backend.py:492), so nothing else is
            # affected by the change in aliasing.
            self.R.copy_(R.detach() if R.requires_grad else R)
            self.T.copy_(t.detach() if t.requires_grad else t)
            return
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
