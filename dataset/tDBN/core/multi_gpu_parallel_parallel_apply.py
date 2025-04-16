def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(
        replicas)])
