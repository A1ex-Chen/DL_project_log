def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[
    torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions:
    bool=False, use_cache: bool=False) ->Tuple[torch.Tensor, Optional[torch
    .Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            'Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.'
            )
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.
        num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.
        num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.
        num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states,
        key_states, cos, sin, position_ids)
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    past_key_value = (key_states, value_states) if use_cache else None
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)
    key_padding_mask = attention_mask
    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=
            torch.int32, device=qkv.device)
        max_s = q_len
        output = flash_attn_unpadded_qkvpacked_func(qkv, cu_q_lens, max_s, 
            0.0, softmax_scale=None, causal=True)
        output = output.view(bsz, q_len, -1)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_unpadded_qkvpacked_func(qkv, cu_q_lens,
            max_s, 0.0, softmax_scale=None, causal=True)
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)
    return self.o_proj(output), None, past_key_value
