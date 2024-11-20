@add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format(
    '(batch_size, sequence_length)'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'layoutlm-base-uncased', output_type=TokenClassifierOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, bbox=None, attention_mask=None,
    token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=
    None, labels=None, output_attentions=None, output_hidden_states=None,
    return_dict=None):
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=
        attention_mask, token_type_ids=token_type_ids, position_ids=
        position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    if not return_dict:
        output = (logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=
        outputs.hidden_states, attentions=outputs.attentions)
