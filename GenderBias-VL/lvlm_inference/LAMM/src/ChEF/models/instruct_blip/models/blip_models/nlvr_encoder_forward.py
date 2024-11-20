def forward(self, input_ids=None, attention_mask=None, position_ids=None,
    head_mask=None, inputs_embeds=None, encoder_embeds=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    past_key_values=None, use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None, is_decoder=False, mode=
    'multimodal'):
    """
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if is_decoder:
        use_cache = (use_cache if use_cache is not None else self.config.
            use_cache)
    else:
        use_cache = False
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'
            )
    elif input_ids is not None:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        device = inputs_embeds.device
    elif encoder_embeds is not None:
        input_shape = encoder_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        device = encoder_embeds.device
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds or encoder_embeds'
            )
    past_key_values_length = past_key_values[0][0].shape[2
        ] if past_key_values is not None else 0
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length +
            past_key_values_length), device=device)
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
        attention_mask, input_shape, device, is_decoder)
    if encoder_hidden_states is not None:
        if type(encoder_hidden_states) == list:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states[0].size())
        else:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size())
        encoder_hidden_shape = encoder_batch_size, encoder_sequence_length
        if type(encoder_attention_mask) == list:
            encoder_extended_attention_mask = [self.invert_attention_mask(
                mask) for mask in encoder_attention_mask]
        elif encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape,
                device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    if encoder_embeds is None:
        embedding_output = self.embeddings(input_ids=input_ids,
            position_ids=position_ids, inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length)
    else:
        embedding_output = encoder_embeds
    encoder_outputs = self.encoder(embedding_output, attention_mask=
        extended_attention_mask, head_mask=head_mask, encoder_hidden_states
        =encoder_hidden_states, encoder_attention_mask=
        encoder_extended_attention_mask, past_key_values=past_key_values,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        mode=mode)
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output
        ) if self.pooler is not None else None
    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]
    return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        sequence_output, pooler_output=pooled_output, past_key_values=
        encoder_outputs.past_key_values, hidden_states=encoder_outputs.
        hidden_states, attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions)
