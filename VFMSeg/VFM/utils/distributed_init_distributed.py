def init_distributed(opt, gpu_index):
    opt['CUDA'] = opt.get('CUDA', True) and torch.cuda.is_available()
    if 'OMPI_COMM_WORLD_SIZE' not in os.environ:
        opt['env_info'] = 'no MPI'
        opt['world_size'] = 1
        opt['local_size'] = 1
        opt['rank'] = 0
        opt['local_rank'] = gpu_index
        opt['master_address'] = '127.0.0.1'
        opt['master_port'] = '8673'
    else:
        opt['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        opt['local_size'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        opt['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        opt['local_rank'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if not opt['CUDA']:
        assert opt['world_size'
            ] == 1, 'multi-GPU training without CUDA is not supported since we use NCCL as communication backend'
        opt['device'] = torch.device('cpu')
    else:
        torch.cuda.set_device(opt['local_rank'])
        opt['device'] = torch.device('cuda', opt['local_rank'])
    return opt
