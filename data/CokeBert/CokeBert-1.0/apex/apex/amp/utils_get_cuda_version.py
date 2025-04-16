def get_cuda_version():
    return tuple(int(x) for x in torch.version.cuda.split('.'))
