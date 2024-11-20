def forward(self, input_ids: Optional[torch.LongTensor]=None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None,
    attention_mask: Optional[torch.FloatTensor]=None, token_type_ids:
    Optional[torch.LongTensor]=None, position_ids: Optional[torch.
    LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None,
    inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states:
    Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.
    FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache:
    Optional[bool]=None, output_attentions: Optional[bool]=None,
    output_hidden_states: Optional[bool]=None, images=None, return_dict:
    Optional[bool]=None) ->Union[Tuple, CausalLMOutputWithPast]:
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    (input_ids, modality_indicators, attention_mask, past_key_values,
        inputs_embeds, labels) = (self.prepare_inputs_labels_for_multimodal
        (input_ids, attention_mask, past_key_values, labels, images))
    outputs = self.transformer(input_ids, modality_indicators=
        modality_indicators, past_key_values=past_key_values,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values
        =outputs.past_key_values, hidden_states=outputs.hidden_states,
        attentions=outputs.attentions)
