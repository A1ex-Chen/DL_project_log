@property
@torch_required
def n_gpu(self):
    return self._setup_devices[1]
