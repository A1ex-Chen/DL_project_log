@add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class
    =_CONFIG_FOR_DOC)
def forward(self, input_ids: torch.LongTensor=None, attention_mask:
    Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=
    None, past_key_values: Optional[List[torch.FloatTensor]]=None,
    inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch
    .LongTensor]=None, use_cache: Optional[bool]=None, output_attentions:
    Optional[bool]=None, output_hidden_states: Optional[bool]=None,
    return_dict: Optional[bool]=None, **kwargs) ->Union[Tuple,
    CausalLMOutputWithPast]:
    """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
    samples = kwargs.get('samples', None)
    if samples:
        if samples['data_type'][0] == 'text':
            has_img = False
        elif samples['data_type'][0] == 'multi':
            has_img = True
        else:
            raise NotImplementedError
        text = samples['text_input']
        if has_img:
            image = samples['image']
            to_regress_embeds, attention_mask, targets, im_mask = (self.
                interleav_wrap(image, text))
        else:
            to_regress_tokens, targets = self.text2emb(text, add_special=True)
            to_regress_embeds = self.model.tok_embeddings(to_regress_tokens
                .input_ids)
            attention_mask = to_regress_tokens.attention_mask
            im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()
        inputs_embeds = to_regress_embeds[:, :self.max_length]
        attention_mask = attention_mask[:, :self.max_length]
        targets = targets[:, :self.max_length]
        im_mask = im_mask[:, :self.max_length].bool()
        labels = targets
    else:
        im_mask = kwargs.get('im_mask', None)
        if im_mask is None and inputs_embeds is not None:
            im_mask = torch.zeros(inputs_embeds.shape[:2]).to(inputs_embeds
                .device)
            im_mask = im_mask.bool()
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions
        =output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, im_mask=im_mask)
    hidden_states = outputs[0]
    logits = self.output(hidden_states)
    logits = logits.float()
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
