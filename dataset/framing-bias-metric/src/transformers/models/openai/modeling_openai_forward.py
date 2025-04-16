@add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'openai-gpt', output_type=SequenceClassifierOutput, config_class=
    _CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
    output_attentions=None, output_hidden_states=None, return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    transformer_outputs = self.transformer(input_ids, attention_mask=
        attention_mask, token_type_ids=token_type_ids, position_ids=
        position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)
    if input_ids is not None:
        batch_size, sequence_length = input_ids.shape[:2]
    else:
        batch_size, sequence_length = inputs_embeds.shape[:2]
    assert self.config.pad_token_id is not None or batch_size == 1, 'Cannot handle batch sizes > 1 if no padding token is defined.'
    if self.config.pad_token_id is None:
        sequence_lengths = -1
    elif input_ids is not None:
        sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1
            ) - 1
    else:
        sequence_lengths = -1
        logger.warning(
            f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjuction with `inputs_embeds.`'
            )
    pooled_logits = logits[range(batch_size), sequence_lengths]
    loss = None
    if labels is not None:
        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).
                view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels
                .view(-1))
    if not return_dict:
        output = (pooled_logits,) + transformer_outputs[1:]
        return (loss,) + output if loss is not None else output
    return SequenceClassifierOutput(loss=loss, logits=pooled_logits,
        hidden_states=transformer_outputs.hidden_states, attentions=
        transformer_outputs.attentions)
