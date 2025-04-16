def __repr__(self):
    return (
        'ContextInfo(invocations={:d}, size_bytes={:d}, run_time_ms={:.4f})'
        .format(self.invocations, self.size_bytes, self.run_time_ms))
