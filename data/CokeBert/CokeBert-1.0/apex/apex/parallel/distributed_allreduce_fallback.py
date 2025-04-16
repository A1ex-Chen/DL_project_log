def allreduce_fallback(self):
    grads = [param.grad.data for param in self.module.parameters() if param
        .grad is not None]
    split_buckets = split_half_float_double(grads)
    if self.retain_allreduce_buffers:
        self.allreduce_buffers = [None for _ in range(len(split_buckets))]
    for i, bucket in enumerate(split_buckets):
        allreduced = self.allreduce_maybe_retain(bucket, i)
