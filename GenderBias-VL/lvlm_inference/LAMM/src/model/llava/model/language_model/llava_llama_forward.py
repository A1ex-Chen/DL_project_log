def forward(self, input_ids: torch.LongTensor=None, attention_mask:
    Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=
    None, past_key_values: Optional[List[torch.FloatTensor]]=None,
    inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch
    .LongTensor]=None, use_cache: Optional[bool]=None, output_attentions:
    Optional[bool]=None, output_hidden_states: Optional[bool]=None, images:
    Optional[torch.FloatTensor]=None, return_dict: Optional[bool]=None
    ) ->Union[Tuple, CausalLMOutputWithPast]:
    if inputs_embeds is None:
        (input_ids, position_ids, attention_mask, past_key_values,
            inputs_embeds, labels) = (self.
            prepare_inputs_labels_for_multimodal(input_ids, position_ids,
            attention_mask, past_key_values, labels, images))
    return super().forward(input_ids=input_ids, attention_mask=
        attention_mask, position_ids=position_ids, past_key_values=
        past_key_values, inputs_embeds=inputs_embeds, labels=labels,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)