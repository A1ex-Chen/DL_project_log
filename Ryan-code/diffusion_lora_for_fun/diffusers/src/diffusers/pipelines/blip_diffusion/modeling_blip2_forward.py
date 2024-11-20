def forward(self, text_input=None, image_input=None, head_mask=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    past_key_values=None, use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    """
        encoder_hidden_states  (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
    text = self.tokenizer(text_input, return_tensors='pt', padding=True)
    text = text.to(self.device)
    input_ids = text.input_ids
    batch_size = input_ids.shape[0]
    query_atts = torch.ones((batch_size, self.query_tokens.size()[1]),
        dtype=torch.long).to(self.device)
    attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    past_key_values_length = past_key_values[0][0].shape[2
        ] - self.config.query_length if past_key_values is not None else 0
    query_length = self.query_tokens.shape[1]
    embedding_output = self.embeddings(input_ids=input_ids, query_embeds=
        self.query_tokens, past_key_values_length=past_key_values_length)
    input_shape = embedding_output.size()[:-1]
    batch_size, seq_length = input_shape
    device = embedding_output.device
    image_embeds_frozen = self.visual_encoder(image_input).last_hidden_state
    encoder_hidden_states = image_embeds_frozen
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length +
            past_key_values_length), device=device)
    extended_attention_mask = self.get_extended_attention_mask(attention_mask,
        input_shape, device)
    if encoder_hidden_states is not None:
        if isinstance(encoder_hidden_states, list):
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states[0].size())
        else:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size())
        encoder_hidden_shape = encoder_batch_size, encoder_sequence_length
        if isinstance(encoder_attention_mask, list):
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
    head_mask = self.get_head_mask(head_mask, self.config.qformer_config.
        num_hidden_layers)
    encoder_outputs = self.encoder(embedding_output, attention_mask=
        extended_attention_mask, head_mask=head_mask, encoder_hidden_states
        =encoder_hidden_states, encoder_attention_mask=
        encoder_extended_attention_mask, past_key_values=past_key_values,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        query_length=query_length)
    sequence_output = encoder_outputs[0]
    pooled_output = sequence_output[:, 0, :]
    if not return_dict:
        return self.proj_layer(sequence_output[:, :query_length, :])
    return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        sequence_output, pooler_output=pooled_output, past_key_values=
        encoder_outputs.past_key_values, hidden_states=encoder_outputs.
        hidden_states, attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions)
