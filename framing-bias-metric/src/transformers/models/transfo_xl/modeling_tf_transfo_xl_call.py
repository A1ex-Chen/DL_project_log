@add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'transfo-xl-wt103', output_type=TFTransfoXLLMHeadModelOutput,
    config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, mems=None, head_mask=None, inputs_embeds=
    None, output_attentions=None, output_hidden_states=None, return_dict=
    None, labels=None, training=False, **kwargs):
    inputs = input_processing(func=self.call, input_ids=input_ids, mems=
        mems, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict, training=training,
        kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.transformer.return_dict
    if inputs['input_ids'] is not None:
        bsz, tgt_len = shape_list(inputs['input_ids'])[:2]
    else:
        bsz, tgt_len = shape_list(inputs['inputs_embeds'])[:2]
    transformer_outputs = self.transformer(inputs['input_ids'], inputs[
        'mems'], inputs['head_mask'], inputs['inputs_embeds'], inputs[
        'output_attentions'], inputs['output_hidden_states'], return_dict,
        training=inputs['training'])
    last_hidden = transformer_outputs[0]
    pred_hid = last_hidden[:, -tgt_len:]
    softmax_output = self.crit(pred_hid, labels, training=inputs['training'])
    if not return_dict:
        return (softmax_output,) + transformer_outputs[1:]
    return TFTransfoXLLMHeadModelOutput(prediction_scores=softmax_output,
        mems=transformer_outputs.mems, hidden_states=transformer_outputs.
        hidden_states, attentions=transformer_outputs.attentions)
