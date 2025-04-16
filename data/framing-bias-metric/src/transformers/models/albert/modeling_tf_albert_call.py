@add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format(
    'batch_size, num_choices, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'albert-base-v2', output_type=TFMultipleChoiceModelOutput, config_class
    =_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    labels=None, training=False, **kwargs):
    """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        labels=labels, training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.albert.return_dict
    if inputs['input_ids'] is not None:
        num_choices = shape_list(inputs['input_ids'])[1]
        seq_length = shape_list(inputs['input_ids'])[2]
    else:
        num_choices = shape_list(inputs['inputs_embeds'])[1]
        seq_length = shape_list(inputs['inputs_embeds'])[2]
    flat_input_ids = tf.reshape(inputs['input_ids'], (-1, seq_length)
        ) if inputs['input_ids'] is not None else None
    flat_attention_mask = tf.reshape(inputs['attention_mask'], (-1, seq_length)
        ) if inputs['attention_mask'] is not None else None
    flat_token_type_ids = tf.reshape(inputs['token_type_ids'], (-1, seq_length)
        ) if inputs['token_type_ids'] is not None else None
    flat_position_ids = tf.reshape(position_ids, (-1, seq_length)
        ) if position_ids is not None else None
    flat_inputs_embeds = tf.reshape(inputs['inputs_embeds'], (-1,
        seq_length, shape_list(inputs['inputs_embeds'])[3])) if inputs[
        'inputs_embeds'] is not None else None
    outputs = self.albert(flat_input_ids, flat_attention_mask,
        flat_token_type_ids, flat_position_ids, inputs['head_mask'],
        flat_inputs_embeds, inputs['output_attentions'], inputs[
        'output_hidden_states'], return_dict=return_dict, training=inputs[
        'training'])
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output, training=inputs['training'])
    logits = self.classifier(pooled_output)
    reshaped_logits = tf.reshape(logits, (-1, num_choices))
    loss = None if inputs['labels'] is None else self.compute_loss(inputs[
        'labels'], reshaped_logits)
    if not return_dict:
        output = (reshaped_logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
