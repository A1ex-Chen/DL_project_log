def compute_bias(self, qlen, klen):
    """ Compute binned relative position bias """
    context_position = torch.arange(qlen, dtype=torch.long)[:, None]
    memory_position = torch.arange(klen, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = self._relative_position_bucket(relative_position,
        bidirectional=self.is_bidirectional, num_buckets=self.
        relative_attention_num_buckets)
    rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
    values = self.relative_attention_bias(rp_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(0)
    return values
