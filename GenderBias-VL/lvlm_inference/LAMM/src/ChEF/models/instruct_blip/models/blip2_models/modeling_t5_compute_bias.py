def compute_bias(self, query_length, key_length, device=None):
    """Compute binned relative position bias"""
    if device is None:
        device = self.relative_attention_bias.weight.device
    context_position = torch.arange(query_length, dtype=torch.long, device=
        device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device
        )[None, :]
    relative_position = memory_position - context_position
    relative_position_bucket = self._relative_position_bucket(relative_position
        , bidirectional=not self.is_decoder, num_buckets=self.
        relative_attention_num_buckets, max_distance=self.
        relative_attention_max_distance)
    values = self.relative_attention_bias(relative_position_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(0)
    return values
