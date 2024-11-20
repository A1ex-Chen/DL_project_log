def _test_nccl_worker(rank, num_gpu, dist_url):
    import torch.distributed as dist
    dist.init_process_group(backend='NCCL', init_method=dist_url, rank=rank,
        world_size=num_gpu)
    dist.barrier(device_ids=[rank])
