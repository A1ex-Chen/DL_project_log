@contextmanager
def timer(name, ndigits=2, sync_gpu=True):
    if sync_gpu:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if sync_gpu:
        torch.cuda.synchronize()
    stop = time.time()
    elapsed = round(stop - start, ndigits)
    logging.info(f'TIMER {name} {elapsed}')
