def parallel_apply(self, replicas, device_ids, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, device_ids)
