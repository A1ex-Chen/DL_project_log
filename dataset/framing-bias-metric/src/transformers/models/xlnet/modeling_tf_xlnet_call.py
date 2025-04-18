@add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'xlnet-base-cased', output_type=TFXLNetForQuestionAnsweringSimpleOutput,
    config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, mems=None, perm_mask=
    None, target_mapping=None, token_type_ids=None, input_mask=None,
    head_mask=None, inputs_embeds=None, use_mems=None, output_attentions=
    None, output_hidden_states=None, return_dict=None, start_positions=None,
    end_positions=None, training=False, **kwargs):
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
        attention_mask=attention_mask, mems=mems, perm_mask=perm_mask,
        target_mapping=target_mapping, token_type_ids=token_type_ids,
        input_mask=input_mask, head_mask=head_mask, inputs_embeds=
        inputs_embeds, use_mems=use_mems, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, start_positions=start_positions,
        end_positions=end_positions, training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.transformer.return_dict
    transformer_outputs = self.transformer(input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'], mems=inputs['mems'],
        perm_mask=inputs['perm_mask'], target_mapping=inputs[
        'target_mapping'], token_type_ids=inputs['token_type_ids'],
        input_mask=inputs['input_mask'], head_mask=inputs['head_mask'],
        inputs_embeds=inputs['inputs_embeds'], use_mems=inputs['use_mems'],
        output_attentions=inputs['output_attentions'], output_hidden_states
        =inputs['output_hidden_states'], return_dict=return_dict, training=
        inputs['training'])
    sequence_output = transformer_outputs[0]
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)
    loss = None
    if inputs['start_positions'] is not None and inputs['end_positions'
        ] is not None:
        labels = {'start_position': inputs['start_positions']}
        labels['end_position'] = inputs['end_positions']
        loss = self.compute_loss(labels, (start_logits, end_logits))
    if not return_dict:
        output = (start_logits, end_logits) + transformer_outputs[1:]
        return (loss,) + output if loss is not None else output
    return TFXLNetForQuestionAnsweringSimpleOutput(loss=loss, start_logits=
        start_logits, end_logits=end_logits, mems=transformer_outputs.mems,
        hidden_states=transformer_outputs.hidden_states, attentions=
        transformer_outputs.attentions)
