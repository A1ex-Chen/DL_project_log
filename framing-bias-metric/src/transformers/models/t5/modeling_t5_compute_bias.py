def compute_bias(self, query_length, key_length):
    """ Compute binned relative position bias """
    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position
    relative_position_bucket = self._relative_position_bucket(relative_position
        , bidirectional=not self.is_decoder, num_buckets=self.
        relative_attention_num_buckets)
    relative_position_bucket = relative_position_bucket.to(self.
        relative_attention_bias.weight.device)
    values = self.relative_attention_bias(relative_position_bucket)
    values = values.permute([2, 0, 1]).unsqueeze(0)
    return values
