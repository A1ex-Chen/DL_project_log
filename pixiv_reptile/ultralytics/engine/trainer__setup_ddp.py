def _setup_ddp(self, world_size):
    """Initializes and sets the DistributedDataParallel parameters for training."""
    torch.cuda.set_device(RANK)
    self.device = torch.device('cuda', RANK)
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else
        'gloo', timeout=timedelta(seconds=10800), rank=RANK, world_size=
        world_size)
