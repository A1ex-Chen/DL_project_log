def __init__(self, chunk_size):
    try:
        import amp_C
        MultiTensorApply.available = True
        self.chunk_size = chunk_size
    except ImportError as err:
        MultiTensorApply.available = False
        MultiTensorApply.import_err = err
