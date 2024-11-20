@add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'ctrl', output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, past=None, attention_mask=None,
    token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=
    None, use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, labels=None, training=False, **kwargs):
    """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
    inputs = input_processing(func=self.call, input_ids=input_ids, past=
        past, attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, use_cache=use_cache, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, labels=labels, training=training,
        kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.transformer.return_dict
    transformer_outputs = self.transformer(input_ids=inputs['input_ids'],
        past=inputs['past'], attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids'], position_ids=inputs[
        'position_ids'], head_mask=inputs['head_mask'], inputs_embeds=
        inputs['inputs_embeds'], use_cache=inputs['use_cache'],
        output_attentions=inputs['output_attentions'], output_hidden_states
        =inputs['output_hidden_states'], return_dict=return_dict, training=
        inputs['training'])
    hidden_states = transformer_outputs[0]
    logits = self.lm_head(hidden_states)
    loss = None
    if inputs['labels'] is not None:
        logits = logits[:, :-1]
        labels = inputs['labels'][:, 1:]
        loss = self.compute_loss(labels, logits)
    if not return_dict:
        output = (logits,) + transformer_outputs[1:]
        return (loss,) + output if loss is not None else output
    return TFCausalLMOutputWithPast(loss=loss, logits=logits,
        past_key_values=transformer_outputs.past_key_values, hidden_states=
        transformer_outputs.hidden_states, attentions=transformer_outputs.
        attentions)
