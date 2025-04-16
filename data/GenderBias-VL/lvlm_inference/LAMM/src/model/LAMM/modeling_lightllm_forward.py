def forward(self, input_ids: torch.LongTensor=None, attention_mask:
    Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=
    None, past_key_values: Optional[List[torch.FloatTensor]]=None,
    inputs_embeds: Optional[torch.FloatTensor]=None, query_embeds: Optional
    [torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None,
    use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None,
    output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None
    ) ->Union[Tuple, CausalLMOutputWithPast]:
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

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\\nI'm not consciours, but I can talk to you."
        ```"""
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.model.base_model.forward(input_ids=input_ids, b_loc=self
        .b_loc, b_start_loc=self.b_start_loc, b_seq_len=self.b_seq_len,
        input_embs=inputs_embeds, **self.infer_state)
    logits = outputs.unsqueeze(1)
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
        =None, hidden_states=None, attentions=None)
