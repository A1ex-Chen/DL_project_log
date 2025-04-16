@add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=TFBaseModelOutput, config_class=
    _CONFIG_FOR_DOC)
def call(self, input_ids, attention_mask=None, head_mask=None,
    inputs_embeds=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, training=False, **kwargs):
    """
        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, TFT5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = TFT5EncoderModel.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids  # Batch size 1
            >>> outputs = model(input_ids)


        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=
        inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        training=training, kwargs_call=kwargs)
    output_attentions = inputs['output_attentions'
        ] if output_attentions else self.config.output_attentions
    output_hidden_states = inputs['output_hidden_states'
        ] if output_hidden_states else self.config.output_hidden_states
    return_dict = return_dict if inputs['return_dict'
        ] is not None else self.config.return_dict
    encoder_outputs = self.encoder(input_ids, attention_mask=inputs[
        'attention_mask'], encoder_hidden_states=None,
        encoder_attention_mask=None, inputs_embeds=inputs['inputs_embeds'],
        head_mask=head_mask, past_key_values=None, use_cache=False,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, training=inputs['training'])
    if not return_dict:
        return encoder_outputs
    if not cast_bool_to_primitive(output_hidden_states, self.config.
        output_hidden_states):
        encoder_outputs = encoder_outputs[:1] + (None,) + encoder_outputs[1:]
    if not cast_bool_to_primitive(output_attentions, self.config.
        output_attentions):
        encoder_outputs = encoder_outputs + (None,)
    return TFBaseModelOutput(last_hidden_state=encoder_outputs[0],
        hidden_states=encoder_outputs[1], attentions=encoder_outputs[2])
