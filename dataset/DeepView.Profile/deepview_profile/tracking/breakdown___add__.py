def __add__(self, other):
    return ContextInfo(size_bytes=self.size_bytes + other.size_bytes,
        run_time_ms=self.run_time_ms + other.run_time_ms, invocations=self.
        invocations + other.invocations)
