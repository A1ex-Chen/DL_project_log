@add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'google/mobilebert-uncased', output_type=TokenClassifierOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
    output_attentions=None, output_hidden_states=None, return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.mobilebert(input_ids, attention_mask=attention_mask,
        token_type_ids=token_type_ids, position_ids=position_ids, head_mask
        =head_mask, inputs_embeds=inputs_embeds, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(active_loss, labels.view(-1), torch
                .tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    if not return_dict:
        output = (logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=
        outputs.hidden_states, attentions=outputs.attentions)
