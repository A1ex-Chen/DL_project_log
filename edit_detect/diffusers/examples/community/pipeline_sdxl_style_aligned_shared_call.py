def shared_call(self, attn: Attention, hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor]=None, attention_mask:
    Optional[torch.Tensor]=None, **kwargs):
    residual = hidden_states
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width
            ).transpose(1, 2)
    batch_size, sequence_length, _ = (hidden_states.shape if 
        encoder_hidden_states is None else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask,
            sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1,
            attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)
            ).transpose(1, 2)
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if self.adain_queries:
        query = adain(query)
    if self.adain_keys:
        key = adain(key)
    if self.adain_values:
        value = adain(value)
    if self.share_attention:
        key = concat_first(key, -2, scale=self.shared_score_scale)
        value = concat_first(value, -2)
        if self.shared_score_shift != 0:
            hidden_states = self.shifted_scaled_dot_product_attention(attn,
                query, key, value)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key,
                value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
    else:
        hidden_states = F.scaled_dot_product_attention(query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, 
        attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size,
            channel, height, width)
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    hidden_states = hidden_states / attn.rescale_output_factor
    return hidden_states
