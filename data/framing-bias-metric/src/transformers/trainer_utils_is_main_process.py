def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]
