def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size == 1:
        return

    def _send_and_wait(r):
        if rank == r:
            tensor = torch.tensor(0, device='cuda')
        else:
            tensor = torch.tensor(1, device='cuda')
        dist.broadcast(tensor, r)
        while tensor.item() == 1:
            time.sleep(1)
    _send_and_wait(0)
    _send_and_wait(1)
