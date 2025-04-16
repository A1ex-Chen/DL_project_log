def barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()
