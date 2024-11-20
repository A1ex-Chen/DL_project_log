def _aggregate_and_get_max_attention_per_token(self, indices: List[int]):
    """Aggregates the attention for each token and computes the max activation value for each token to alter."""
    attention_maps = self.attention_store.aggregate_attention(from_where=(
        'up', 'down', 'mid'))
    max_attention_per_index = self._compute_max_attention_per_index(
        attention_maps=attention_maps, indices=indices)
    return max_attention_per_index
