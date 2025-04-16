@add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=TFOpenAIGPTDoubleHeadsModelOutput,
    config_class=_CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=
    None, output_attentions=None, output_hidden_states=None, return_dict=
    None, training=False, **kwargs):
    """
        mc_token_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1]``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel

            >>> tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            >>> model = TFOpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            >>> model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
            >>> print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoding = tokenizer(choices, return_tensors="tf")
            >>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
            >>> inputs["mc_token_ids"]= tf.constant([inputs["input_ids"].shape[-1] - 1, inputs["input_ids"].shape[-1] - 1])[None, :]  # Batch size 1
            >>> outputs = model(inputs)
            >>> lm_prediction_scores, mc_prediction_scores = outputs[:2]
        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, mc_token_ids=mc_token_ids, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.transformer.return_dict
    if inputs['input_ids'] is not None:
        input_shapes = shape_list(inputs['input_ids'])
    else:
        input_shapes = shape_list(inputs['inputs_embeds'])[:-1]
    seq_length = input_shapes[-1]
    flat_input_ids = tf.reshape(inputs['input_ids'], (-1, seq_length)
        ) if inputs['input_ids'] is not None else None
    flat_attention_mask = tf.reshape(inputs['attention_mask'], (-1, seq_length)
        ) if inputs['attention_mask'] is not None else None
    flat_token_type_ids = tf.reshape(inputs['token_type_ids'], (-1, seq_length)
        ) if inputs['token_type_ids'] is not None else None
    flat_position_ids = tf.reshape(inputs['position_ids'], (-1, seq_length)
        ) if inputs['position_ids'] is not None else None
    transformer_outputs = self.transformer(flat_input_ids,
        flat_attention_mask, flat_token_type_ids, flat_position_ids, inputs
        ['head_mask'], inputs['inputs_embeds'], inputs['output_attentions'],
        inputs['output_hidden_states'], return_dict=return_dict, training=
        inputs['training'])
    hidden_states = transformer_outputs[0]
    hidden_states = tf.reshape(hidden_states, input_shapes + shape_list(
        hidden_states)[-1:])
    lm_logits = self.transformer.tokens_embed(hidden_states, mode='linear')
    mc_logits = self.multiple_choice_head(hidden_states, inputs[
        'mc_token_ids'], training=inputs['training'])
    mc_logits = tf.squeeze(mc_logits, axis=-1)
    if not return_dict:
        return (lm_logits, mc_logits) + transformer_outputs[1:]
    return TFOpenAIGPTDoubleHeadsModelOutput(logits=lm_logits, mc_logits=
        mc_logits, hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions)
