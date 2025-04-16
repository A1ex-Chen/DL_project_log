def forward(self, input_ids: torch.LongTensor, past_key_values: Optional[
    List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.
    ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None,
    sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.
    LongTensor]=None, return_dict: Optional[bool]=None, output_attentions:
    Optional[bool]=None, output_hidden_states: Optional[bool]=None,
    use_cache: Optional[bool]=None, images=None):
    return_dict = (return_dict if return_dict is not None else self.config.
        return_dict)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    (input_ids, _, attention_mask, past_key_values, inputs_embeds, labels) = (
        self.prepare_inputs_labels_for_multimodal(input_ids, None,
        attention_mask, past_key_values, labels, images))
    outputs = self.transformer(input_ids=input_ids, inputs_embeds=
        inputs_embeds, past_key_values=past_key_values, attention_mask=
        attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id,
        return_dict=return_dict, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, use_cache=use_cache)
    logits = F.linear(outputs.last_hidden_state.to(self.transformer.wte.
        weight.device), self.transformer.wte.weight)
    if self.logit_scale is not None:
        if self.logit_scale == 0:
            warnings.warn(
                f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.'
                )
        logits *= self.logit_scale
    loss = None
    if labels is not None:
        labels = torch.roll(labels, shifts=-1)
        labels[:, -1] = -100
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(
            logits.device).view(-1))
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values
        =outputs.past_key_values, hidden_states=outputs.hidden_states)
