def xformers_forward(self, hidden_states: torch.Tensor, attention_mask:
    Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=
    None, past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False, use_cache: bool=False) ->Tuple[torch.
    Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.
        num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
        self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.
        num_heads, self.head_dim).transpose(1, 2)
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
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            attn_output = xformers.ops.memory_efficient_attention(query_states,
                key_states, value_states, attn_bias=None)
        else:
            attn_output = xformers.ops.memory_efficient_attention(query_states,
                key_states, value_states, attn_bias=xformers.ops.
                LowerTriangularMask())
        attn_weights = None
    else:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {bsz * self.num_heads, q_len, kv_seq_len}, but is {attn_weights.size()}'
                )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.size()}'
                    )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo
                (attn_weights.dtype).min))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=
            torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.size()}'
                )
        attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value
