def init_distributed(gpu_id, global_rank, world_size, dist_url, dist_backend):
    torch.cuda.set_device(gpu_id)
    print(
        f'| distributed init (rank {global_rank}) (world {world_size}): {dist_url}'
        , flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=
        dist_url, world_size=world_size, rank=global_rank)
    torch.distributed.barrier()
    setup_print_for_distributed(is_primary())
