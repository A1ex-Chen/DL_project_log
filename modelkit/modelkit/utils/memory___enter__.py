def __enter__(self):
    self.start_time = time.perf_counter()
    if platform.system() == 'Windows':
        return self
    self.pre_maxrss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return self
