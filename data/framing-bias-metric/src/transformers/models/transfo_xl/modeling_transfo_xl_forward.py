@add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'transfo-xl-wt103', output_type=TransfoXLLMHeadModelOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, mems=None, head_mask=None, inputs_embeds=
    None, labels=None, output_attentions=None, output_hidden_states=None,
    return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if input_ids is not None:
        bsz, tgt_len = input_ids.size(0), input_ids.size(1)
    elif inputs_embeds is not None:
        bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')
    transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=
        head_mask, inputs_embeds=inputs_embeds, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    last_hidden = transformer_outputs[0]
    pred_hid = last_hidden[:, -tgt_len:]
    softmax_output = self.crit(pred_hid, labels)
    prediction_scores = softmax_output.view(bsz, tgt_len, -1
        ) if labels is None else ()
    loss = softmax_output.view(bsz, tgt_len - 1
        ) if labels is not None else None
    if not return_dict:
        output = (prediction_scores,) + transformer_outputs[1:]
        return (loss,) + output if loss is not None else output
    return TransfoXLLMHeadModelOutput(losses=loss, prediction_scores=
        prediction_scores, mems=transformer_outputs.mems, hidden_states=
        transformer_outputs.hidden_states, attentions=transformer_outputs.
        attentions)
