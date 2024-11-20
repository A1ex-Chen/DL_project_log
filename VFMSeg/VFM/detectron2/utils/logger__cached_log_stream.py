@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = PathManager.open(filename, 'a', buffering=1024 if '://' in
        filename else -1)
    atexit.register(io.close)
    return io
