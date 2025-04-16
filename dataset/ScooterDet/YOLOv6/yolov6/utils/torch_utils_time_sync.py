def time_sync():
    """Waits for all kernels in all streams on a CUDA device to complete if cuda is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
