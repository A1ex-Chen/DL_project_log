def is_local_process_zero(self) ->bool:
    """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
    if is_torch_tpu_available():
        return xm.is_master_ordinal(local=True)
    else:
        return self.args.local_rank in [-1, 0]
