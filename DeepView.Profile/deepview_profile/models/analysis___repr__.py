def __repr__(self):
    return 'PerformanceLimits(max_batch={:.2f}, thpt_limit={:.2f})'.format(self
        .max_batch_size, self.throughput_limit)
