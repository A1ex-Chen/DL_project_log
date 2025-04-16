@add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format(
    'batch_size, num_choicec, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'xlm-mlm-en-2048', output_type=MultipleChoiceModelOutput, config_class=
    _CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, langs=None,
    token_type_ids=None, position_ids=None, lengths=None, cache=None,
    head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    num_choices = input_ids.shape[1
        ] if input_ids is not None else inputs_embeds.shape[1]
    input_ids = input_ids.view(-1, input_ids.size(-1)
        ) if input_ids is not None else None
    attention_mask = attention_mask.view(-1, attention_mask.size(-1)
        ) if attention_mask is not None else None
    token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)
        ) if token_type_ids is not None else None
    position_ids = position_ids.view(-1, position_ids.size(-1)
        ) if position_ids is not None else None
    langs = langs.view(-1, langs.size(-1)) if langs is not None else None
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2),
        inputs_embeds.size(-1)) if inputs_embeds is not None else None
    if lengths is not None:
        logger.warn(
            'The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the attention mask instead.'
            )
        lengths = None
    transformer_outputs = self.transformer(input_ids=input_ids,
        attention_mask=attention_mask, langs=langs, token_type_ids=
        token_type_ids, position_ids=position_ids, lengths=lengths, cache=
        cache, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    output = transformer_outputs[0]
    logits = self.sequence_summary(output)
    logits = self.logits_proj(logits)
    reshaped_logits = logits.view(-1, num_choices)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
    if not return_dict:
        output = (reshaped_logits,) + transformer_outputs[1:]
        return (loss,) + output if loss is not None else output
    return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits,
        hidden_states=transformer_outputs.hidden_states, attentions=
        transformer_outputs.attentions)
