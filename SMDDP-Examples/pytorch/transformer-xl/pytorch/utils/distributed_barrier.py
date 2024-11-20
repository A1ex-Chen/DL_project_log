def barrier():
    """
    Call dist.barrier() if distritubed is in use
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
