def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz
    ):
    if 'prev_key' in saved_state:
        _prev_key = saved_state['prev_key']
        assert _prev_key is not None
        prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
        if static_kv:
            k = prev_key
        else:
            assert k is not None
            k = torch.cat([prev_key, k], dim=1)
    if 'prev_value' in saved_state:
        _prev_value = saved_state['prev_value']
        assert _prev_value is not None
        prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
        if static_kv:
            v = prev_value
        else:
            assert v is not None
            v = torch.cat([prev_value, v], dim=1)
    assert k is not None and v is not None
    prev_key_padding_mask: Optional[Tensor] = saved_state.get(
        'prev_key_padding_mask', None)
    if prev_key_padding_mask is not None:
        if static_kv:
            new_key_padding_mask = prev_key_padding_mask
        else:
            new_key_padding_mask = torch.cat([prev_key_padding_mask,
                key_padding_mask], dim=1)
    else:
        new_key_padding_mask = key_padding_mask
    return k, v, new_key_padding_mask
