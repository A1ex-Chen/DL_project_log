def forward(self, hidden_states: torch.Tensor, modality_indicators: torch.
    Tensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids:
    Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.
    Tensor]]=None, output_attentions: Optional[bool]=False, use_cache:
    Optional[bool]=False) ->Tuple[torch.FloatTensor, Optional[Tuple[torch.
    FloatTensor, torch.FloatTensor]]]:
    """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states, modality_indicators)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states, modality_indicators=
        modality_indicators, attention_mask=attention_mask, position_ids=
        position_ids, past_key_value=past_key_value, output_attentions=
        output_attentions, use_cache=use_cache)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states,
        modality_indicators)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    outputs = hidden_states,
    if output_attentions:
        outputs += self_attn_weights,
    if use_cache:
        outputs += present_key_value,
    return outputs
