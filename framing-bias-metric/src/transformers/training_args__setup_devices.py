@cached_property
@torch_required
def _setup_devices(self) ->Tuple['torch.device', int]:
    logger.info('PyTorch: setting up devices')
    if self.no_cuda:
        device = torch.device('cpu')
        n_gpu = 0
    elif is_torch_tpu_available():
        device = xm.xla_device()
        n_gpu = 0
    elif self.local_rank == -1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', self.local_rank)
        n_gpu = 1
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device, n_gpu
