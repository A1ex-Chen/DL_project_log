def sync_bucket_structure(self):
    for tmp_bucket in self.tmp_buckets:
        if len(tmp_bucket) > 0:
            self.active_i_buckets.append(tmp_bucket)
    self.num_buckets = len(self.active_i_buckets)
    self.bucket_sizes = [len(bucket) for bucket in self.active_i_buckets]
    info_tensor = torch.cuda.IntTensor([self.num_buckets] + self.
        bucket_sizes + list(chain(*self.active_i_buckets)))
    dist.broadcast(info_tensor, 0)
    info = [int(entry) for entry in info_tensor]
    self.num_buckets = info[0]
    self.bucket_sizes = info[1:self.num_buckets + 1]
    self.buckets = [[None for _ in range(self.bucket_sizes[i])] for i in
        range(self.num_buckets)]
    self.active_i_buckets = [[None for _ in range(self.bucket_sizes[i])] for
        i in range(self.num_buckets)]
    flattened_buckets = info[self.num_buckets + 1:]
    flat_i = 0
    for bucket_idx in range(self.num_buckets):
        for bucket_loc in range(self.bucket_sizes[bucket_idx]):
            param_i = flattened_buckets[flat_i]
            self.active_i_buckets[bucket_idx][bucket_loc] = param_i
            self.param_id_to_bucket[id(self.active_params[param_i])
                ] = bucket_idx, bucket_loc
            flat_i += 1
