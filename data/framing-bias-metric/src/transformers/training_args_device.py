@property
@torch_required
def device(self) ->'torch.device':
    """
        The device used by this process.
        """
    return self._setup_devices[0]
