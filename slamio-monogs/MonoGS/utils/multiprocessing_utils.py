import copy
import queue

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


class InlineBackendQueue:
    """
    Ported from RTGS (github.com/UMN-ZhaoLab/RTGS, MonoGS_fullend branch,
    multiprocessing_utils.py). Used as the backend_queue when
    use_inline_backend is enabled (see slam.py): .put() calls straight into
    BackEnd.process_message() synchronously, in the same call stack as
    whoever queued it - no separate process or thread, so frontend and
    backend share one CUDA context instead of each getting their own via
    mp.Process. Trades away the continuous background-mapping loop in
    slam_backend.py's run() (which never executes here, since nothing ever
    drives that loop) for eliminating cross-process CUDA context-switch
    overhead - same trade-off RTGS's own tested pipeline makes.
    """

    def __init__(self):
        self.backend = None

    def set_backend(self, backend):
        self.backend = backend

    def put(self, data):
        if self.backend is None:
            raise RuntimeError("InlineBackendQueue backend not set")
        self.backend.process_message(data)

    def get(self):
        raise queue.Empty

    def empty(self):
        return True


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
