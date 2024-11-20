def forward(self, input_ids: torch.LongTensor=None, attention_mask:
    Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.
    FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None,
    labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None,
    output_attentions: Optional[bool]=None, output_hidden_states: Optional[
    bool]=None, images: Optional[torch.FloatTensor]=None, return_dict:
    Optional[bool]=None) ->Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        images=images)
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
