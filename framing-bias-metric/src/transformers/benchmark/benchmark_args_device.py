@property
@torch_required
def device(self) ->'torch.device':
    return self._setup_devices[0]
