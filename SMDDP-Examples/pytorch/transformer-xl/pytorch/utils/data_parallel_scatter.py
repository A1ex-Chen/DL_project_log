def scatter(self, inputs, kwargs, device_ids):
    bsz = inputs[0].size(self.dim)
    num_dev = len(self.device_ids)
    gpu0_bsz = self.gpu0_bsz
    bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
    if gpu0_bsz < bsz_unit:
        chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
        delta = bsz - sum(chunk_sizes)
        for i in range(delta):
            chunk_sizes[i + 1] += 1
        if gpu0_bsz == 0:
            chunk_sizes = chunk_sizes[1:]
    else:
        return super().scatter(inputs, kwargs, device_ids)
    return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim
        )
