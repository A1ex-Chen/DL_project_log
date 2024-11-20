def __iter__(self):
    g = torch.Generator()
    g.manual_seed(self.epoch)
    global_bsz = self.global_batch_size
    indices = []
    for bid in range(self.num_buckets):
        perm = torch.randperm(len(self.buckets[bid]), generator=g)
        bucket_indices = self.buckets[bid][perm]
        indices.append(bucket_indices)
    indices = torch.cat(indices)
    length = len(indices) // global_bsz * global_bsz
    indices = indices[:length]
    assert len(indices) % self.global_batch_size == 0
    indices = self.reshuffle_batches(indices, g)
    indices = self.distribute_batches(indices)
    return iter(indices)
