def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True
