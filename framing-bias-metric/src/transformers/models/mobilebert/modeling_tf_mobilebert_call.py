@add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'google/mobilebert-uncased', output_type=TFTokenClassifierOutput,
    config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    labels=None, training=False, **kwargs):
    """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        labels=labels, training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.mobilebert.return_dict
    outputs = self.mobilebert(inputs['input_ids'], attention_mask=inputs[
        'attention_mask'], token_type_ids=inputs['token_type_ids'],
        position_ids=inputs['position_ids'], head_mask=inputs['head_mask'],
        inputs_embeds=inputs['inputs_embeds'], output_attentions=inputs[
        'output_attentions'], output_hidden_states=inputs[
        'output_hidden_states'], return_dict=return_dict, training=inputs[
        'training'])
    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output, training=training)
    logits = self.classifier(sequence_output)
    loss = None if inputs['labels'] is None else self.compute_loss(inputs[
        'labels'], logits)
    if not return_dict:
        output = (logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=
        outputs.hidden_states, attentions=outputs.attentions)
