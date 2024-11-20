@contextlib.contextmanager
def switch_on_quantization(do_quantization=False):
    assert not do_quantization, 'quantization is not available'
    try:
        yield
    finally:
        pass
