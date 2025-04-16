def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False
