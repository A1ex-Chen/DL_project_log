@property
@torch_required
def parallel_mode(self):
    """
        The current mode used for parallelism if multiple GPUs/TPU cores are available. One of:

        - :obj:`ParallelMode.NOT_PARALLEL`: no parallelism (CPU or one GPU).
        - :obj:`ParallelMode.NOT_DISTRIBUTED`: several GPUs in one single process (uses :obj:`torch.nn.DataParallel`).
        - :obj:`ParallelMode.DISTRIBUTED`: several GPUs, each ahving its own process (uses
          :obj:`torch.nn.DistributedDataParallel`).
        - :obj:`ParallelMode.TPU`: several TPU cores.
        """
    if is_torch_tpu_available():
        return ParallelMode.TPU
    elif self.local_rank != -1:
        return ParallelMode.DISTRIBUTED
    elif self.n_gpu > 1:
        return ParallelMode.NOT_DISTRIBUTED
    else:
        return ParallelMode.NOT_PARALLEL
