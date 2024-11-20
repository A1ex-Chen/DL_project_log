def init_seeds(seed=0, deterministic=False):
    import torch.backends.cudnn as cudnn
    if deterministic and check_version(torch.__version__, '1.12.0'):
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (
        True, False)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
