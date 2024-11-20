@add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
def forward(self, input_ids: torch.LongTensor=None, attention_mask:
    Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=
    None, past_key_values: Optional[List[torch.FloatTensor]]=None,
    inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[
    bool]=None, output_attentions: Optional[bool]=None,
    output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=
    None, **kwargs) ->Union[Tuple, BaseModelOutputWithPast]:
    im_mask = kwargs.get('im_mask', None)
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'
            )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if position_ids is None:
        device = (input_ids.device if input_ids is not None else
            inputs_embeds.device)
        position_ids = torch.arange(past_key_values_length, seq_length +
            past_key_values_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
    if inputs_embeds is None:
        inputs_embeds = self.tok_embeddings(input_ids)
        im_mask = torch.zeros(inputs_embeds.shape[:2]).to(inputs_embeds.device
            ).bool()
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past),
            dtype=torch.bool, device=inputs_embeds.device)
    attention_mask = self._prepare_decoder_attention_mask(attention_mask, (
        batch_size, seq_length), inputs_embeds, past_key_values_length)
    hidden_states = inputs_embeds
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
            use_cache = False
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += hidden_states,
        past_key_value = past_key_values[idx
            ] if past_key_values is not None else None
        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    return module(*inputs, output_attentions, None, im_mask)
                return custom_forward
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer), hidden_states,
                attention_mask, position_ids, None)
        else:
            layer_outputs = decoder_layer(hidden_states, attention_mask=
                attention_mask, position_ids=position_ids, past_key_value=
                past_key_value, output_attentions=output_attentions,
                use_cache=use_cache, im_mask=im_mask)
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += layer_outputs[2 if output_attentions else 1],
        if output_attentions:
            all_self_attns += layer_outputs[1],
    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += hidden_states,
    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache,
            all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states,
        past_key_values=next_cache, hidden_states=all_hidden_states,
        attentions=all_self_attns)
