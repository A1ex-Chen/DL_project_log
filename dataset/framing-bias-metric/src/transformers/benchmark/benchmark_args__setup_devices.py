@cached_property
@torch_required
def _setup_devices(self) ->Tuple['torch.device', int]:
    logger.info('PyTorch: setting up devices')
    if not self.cuda:
        device = torch.device('cpu')
        n_gpu = 0
    elif is_torch_tpu_available():
        device = xm.xla_device()
        n_gpu = 0
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    return device, n_gpu
