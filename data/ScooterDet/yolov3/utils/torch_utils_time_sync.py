def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
