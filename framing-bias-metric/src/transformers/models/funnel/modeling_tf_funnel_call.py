@add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'funnel-transformer/small', output_type=TFQuestionAnsweringModelOutput,
    config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, token_type_ids=None,
    inputs_embeds=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, start_positions=None, end_positions=None, training=
    False, **kwargs):
    """
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        start_positions=start_positions, end_positions=end_positions,
        training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.funnel.return_dict
    outputs = self.funnel(inputs['input_ids'], inputs['attention_mask'],
        inputs['token_type_ids'], inputs['inputs_embeds'], inputs[
        'output_attentions'], inputs['output_hidden_states'], return_dict=
        return_dict, training=inputs['training'])
    sequence_output = outputs[0]
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)
    loss = None
    if inputs['start_positions'] is not None and inputs['end_positions'
        ] is not None:
        labels = {'start_position': inputs['start_positions'],
            'end_position': inputs['end_positions']}
        loss = self.compute_loss(labels, (start_logits, end_logits))
    if not return_dict:
        output = (start_logits, end_logits) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return TFQuestionAnsweringModelOutput(loss=loss, start_logits=
        start_logits, end_logits=end_logits, hidden_states=outputs.
        hidden_states, attentions=outputs.attentions)
