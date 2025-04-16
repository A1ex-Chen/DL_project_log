@add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format(
    'batch_size, num_choices, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'allenai/longformer-base-4096', output_type=
    LongformerMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
    global_attention_mask=None, labels=None, position_ids=None,
    inputs_embeds=None, output_attentions=None, output_hidden_states=None,
    return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
    num_choices = input_ids.shape[1
        ] if input_ids is not None else inputs_embeds.shape[1]
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if global_attention_mask is None and input_ids is not None:
        logger.info('Initializing global attention on multiple choice...')
        global_attention_mask = torch.stack([_compute_global_attention_mask
            (input_ids[:, i], self.config.sep_token_id, before_sep_token=
            False) for i in range(num_choices)], dim=1)
    flat_input_ids = input_ids.view(-1, input_ids.size(-1)
        ) if input_ids is not None else None
    flat_position_ids = position_ids.view(-1, position_ids.size(-1)
        ) if position_ids is not None else None
    flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)
        ) if token_type_ids is not None else None
    flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)
        ) if attention_mask is not None else None
    flat_global_attention_mask = global_attention_mask.view(-1,
        global_attention_mask.size(-1)
        ) if global_attention_mask is not None else None
    flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2),
        inputs_embeds.size(-1)) if inputs_embeds is not None else None
    outputs = self.longformer(flat_input_ids, position_ids=
        flat_position_ids, token_type_ids=flat_token_type_ids,
        attention_mask=flat_attention_mask, global_attention_mask=
        flat_global_attention_mask, inputs_embeds=flat_inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    reshaped_logits = logits.view(-1, num_choices)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
    if not return_dict:
        output = (reshaped_logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return LongformerMultipleChoiceModelOutput(loss=loss, logits=
        reshaped_logits, hidden_states=outputs.hidden_states, attentions=
        outputs.attentions, global_attentions=outputs.global_attentions)
