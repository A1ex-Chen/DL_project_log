def is_world_process_zero(self) ->bool:
    """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
    if is_torch_tpu_available():
        return xm.is_master_ordinal(local=False)
    else:
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0
