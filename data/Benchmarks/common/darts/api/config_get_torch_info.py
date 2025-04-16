def get_torch_info():
    """Get Pytorch system info"""
    VersionInfo = namedtuple('PytorchVersionInfo',
        'torch_version cuda_version cudnn_version')
    return VersionInfo(torch.__version__, torch.version.cuda, torch.
        backends.cudnn.version())
