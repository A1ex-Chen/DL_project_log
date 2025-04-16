def _concat_saved_state(self, k, v, saved_state, static_kv, bsz) ->Tuple[Tensor
    ]:
    prev_K = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.
        head_dim)
    prev_V = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.
        head_dim)
    new_K = prev_K if static_kv else torch.cat([prev_K, k], dim=1)
    new_V = prev_V if static_kv else torch.cat([prev_V, v], dim=1)
    return new_K, new_V
