def __call__(self, attn: 'Attention', hidden_states, encoder_hidden_states=
    None, attention_mask=None, temb=None):
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    hidden_states = hidden_states.view(hidden_states.shape[0],
        hidden_states.shape[1], -1).transpose(1, 2)
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask,
        sequence_length, batch_size)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)
    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
        1, 2)
    query = attn.to_q(hidden_states)
    dim = query.shape[-1]
    query = attn.head_to_batch_dim(query)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.head_to_batch_dim(
        encoder_hidden_states_key_proj)
    encoder_hidden_states_value_proj = attn.head_to_batch_dim(
        encoder_hidden_states_value_proj)
    if not attn.only_cross_attention:
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
    else:
        key = encoder_hidden_states_key_proj
        value = encoder_hidden_states_value_proj
    batch_size_attention, query_tokens, _ = query.shape
    hidden_states = torch.zeros((batch_size_attention, query_tokens, dim //
        attn.heads), device=query.device, dtype=query.dtype)
    for i in range(batch_size_attention // self.slice_size):
        start_idx = i * self.slice_size
        end_idx = (i + 1) * self.slice_size
        query_slice = query[start_idx:end_idx]
        key_slice = key[start_idx:end_idx]
        attn_mask_slice = attention_mask[start_idx:end_idx
            ] if attention_mask is not None else None
        attn_slice = attn.get_attention_scores(query_slice, key_slice,
            attn_mask_slice)
        attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
        hidden_states[start_idx:end_idx] = attn_slice
    hidden_states = attn.batch_to_head_dim(hidden_states)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
    hidden_states = hidden_states + residual
    return hidden_states
