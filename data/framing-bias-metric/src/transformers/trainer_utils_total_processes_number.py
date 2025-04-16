def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    elif local_rank != -1 and is_torch_available():
        import torch
        return torch.distributed.get_world_size()
    return 1
