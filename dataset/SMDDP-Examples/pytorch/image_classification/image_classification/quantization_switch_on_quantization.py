@contextlib.contextmanager
def switch_on_quantization(do_quantization=True):
    """Context manager for quantization activation"""
    if do_quantization:
        initialize()
    try:
        yield
    finally:
        if do_quantization:
            deactivate()
