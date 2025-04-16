def _hash_vectors(self, vectors, num_hashes, attention_mask,
    increase_num_buckets=False):
    batch_size = vectors.shape[0]
    if isinstance(self.num_buckets, int):
        assert self.num_buckets % 2 == 0, 'There should be an even number of bucktes, but `self.num_bucktes`: {}'.format(
            self.num_buckets)
        rotation_size = self.num_buckets
        num_buckets = self.num_buckets
    else:
        rotation_size, num_buckets = 0, 1
        for bucket_factor in self.num_buckets:
            assert bucket_factor % 2 == 0, 'The number of buckets should be even, but `num_bucket`: {}'.format(
                bucket_factor)
            rotation_size = rotation_size + bucket_factor
            num_buckets = num_buckets * bucket_factor
    vectors = vectors.detach()
    if self.hash_seed is not None:
        torch.manual_seed(self.hash_seed)
    rotations_shape = self.num_attention_heads, vectors.shape[-1
        ], num_hashes, rotation_size // 2
    random_rotations = torch.randn(rotations_shape, device=vectors.device,
        dtype=vectors.dtype)
    rotated_vectors = torch.einsum('bmtd,mdhr->bmhtr', vectors,
        random_rotations)
    if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
        rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1
            )
        buckets = torch.argmax(rotated_vectors, dim=-1)
    else:
        buckets, cur_sum, cur_product = None, 0, 1
        for bucket_factor in self.num_buckets:
            rotated_vectors_factor = rotated_vectors[..., cur_sum:cur_sum +
                bucket_factor // 2]
            cur_sum = cur_sum + bucket_factor // 2
            rotated_vectors_factor = torch.cat([rotated_vectors_factor, -
                rotated_vectors_factor], dim=-1)
            if buckets is None:
                buckets = torch.argmax(rotated_vectors_factor, dim=-1)
            else:
                buckets = buckets + cur_product * torch.argmax(
                    rotated_vectors_factor, dim=-1)
            cur_product = cur_product * bucket_factor
    if attention_mask is not None and attention_mask.sum().item(
        ) < batch_size * attention_mask.shape[-1]:
        num_buckets = num_buckets + 1
        buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(
            buckets.shape)
        buckets = torch.where(buckets_mask, buckets, torch.tensor(
            num_buckets - 1, dtype=torch.long, device=buckets.device))
    elif increase_num_buckets:
        num_buckets = num_buckets + 1
    offsets = torch.arange(num_hashes, device=vectors.device)
    offsets = (offsets * num_buckets).view((1, 1, -1, 1))
    offsets = offsets.expand((batch_size, self.num_attention_heads) +
        offsets.shape[-2:])
    offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
    return offset_buckets
