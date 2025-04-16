def _timed_function(*args, **kwargs):
    start = time.time()
    ret = f(*args, **kwargs)
    return ret, time.time() - start
