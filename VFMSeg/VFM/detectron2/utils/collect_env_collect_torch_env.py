def collect_torch_env():
    try:
        import torch.__config__
        return torch.__config__.show()
    except ImportError:
        from torch.utils.collect_env import get_pretty_env_info
        return get_pretty_env_info()
