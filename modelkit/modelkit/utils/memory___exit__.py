def __exit__(self, *args):
    self.time = time.perf_counter() - self.start_time
    if platform.system() == 'Windows':
        return
    post_maxrss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    self.increment = post_maxrss_bytes - self.pre_maxrss_bytes
