def _set_num_buckets(self, sequence_length):
    num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)
        ).bit_length() - 1
    num_buckets = 2 ** num_buckets_pow_2
    num_buckets_limit = 2 * max(int((self.max_position_embeddings // self.
        chunk_length) ** 0.5), self.chunk_length)
    if num_buckets > num_buckets_limit:
        num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (
            num_buckets_pow_2 - num_buckets_pow_2 // 2)]
    logger.warning(
        'config.num_buckets is not set. Setting config.num_buckets to {}...'
        .format(num_buckets))
    self.config.num_buckets = num_buckets
    self.num_buckets = num_buckets
