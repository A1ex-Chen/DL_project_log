@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        if 'CUDA out of memory. ' in str(e):
            pass
        else:
            raise
