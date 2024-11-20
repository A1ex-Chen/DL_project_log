def test_nccl_ops():
    num_gpu = torch.cuda.device_count()
    if os.access('/tmp', os.W_OK):
        import torch.multiprocessing as mp
        dist_url = 'file:///tmp/nccl_tmp_file'
        print('Testing NCCL connectivity ... this should not hang.')
        mp.spawn(_test_nccl_worker, nprocs=num_gpu, args=(num_gpu, dist_url
            ), daemon=False)
        print('NCCL succeeded.')
