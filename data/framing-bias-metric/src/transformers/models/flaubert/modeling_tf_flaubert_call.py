@add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'jplu/tf-flaubert-small-cased', output_type=
    TFFlaubertWithLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, langs=None,
    token_type_ids=None, position_ids=None, lengths=None, cache=None,
    head_mask=None, inputs_embeds=None, output_attentions=None,
    output_hidden_states=None, return_dict=None, training=False, **kwargs):
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, langs=langs, token_type_ids=
        token_type_ids, position_ids=position_ids, lengths=lengths, cache=
        cache, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict, training=training,
        kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.transformer.return_dict
    transformer_outputs = self.transformer(input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'], langs=inputs['langs'],
        token_type_ids=inputs['token_type_ids'], position_ids=inputs[
        'position_ids'], lengths=inputs['lengths'], cache=inputs['cache'],
        head_mask=inputs['head_mask'], inputs_embeds=inputs['inputs_embeds'
        ], output_attentions=inputs['output_attentions'],
        output_hidden_states=inputs['output_hidden_states'], return_dict=
        return_dict, training=inputs['training'])
    output = transformer_outputs[0]
    outputs = self.pred_layer(output)
    if not return_dict:
        return (outputs,) + transformer_outputs[1:]
    return TFFlaubertWithLMHeadModelOutput(logits=outputs, hidden_states=
        transformer_outputs.hidden_states, attentions=transformer_outputs.
        attentions)
