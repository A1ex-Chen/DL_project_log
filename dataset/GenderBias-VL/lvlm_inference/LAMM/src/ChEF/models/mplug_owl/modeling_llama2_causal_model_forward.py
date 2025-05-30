def causal_model_forward(self, input_ids: torch.LongTensor=None,
    modality_indicators: torch.Tensor=None, attention_mask: Optional[torch.
    Tensor]=None, position_ids: Optional[torch.LongTensor]=None,
    past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds:
    Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=
    None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]
    =None, output_hidden_states: Optional[bool]=None, return_dict: Optional
    [bool]=None) ->Union[Tuple, CausalLMOutputWithPast]:
    """
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.model(input_ids=input_ids, modality_indicators=
        modality_indicators, attention_mask=attention_mask, position_ids=
        position_ids, past_key_values=past_key_values, inputs_embeds=
        inputs_embeds, use_cache=use_cache, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.
            config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range
            (self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
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
