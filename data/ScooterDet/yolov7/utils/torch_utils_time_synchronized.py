def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
