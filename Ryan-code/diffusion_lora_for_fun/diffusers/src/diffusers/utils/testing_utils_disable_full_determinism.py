def disable_full_determinism():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ''
    torch.use_deterministic_algorithms(False)
