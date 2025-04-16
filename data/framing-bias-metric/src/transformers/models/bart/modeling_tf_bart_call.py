@add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=
    _CONFIG_FOR_DOC)
def call(self, input_ids, attention_mask=None, decoder_input_ids=None,
    decoder_attention_mask=None, encoder_outputs: Optional[
    TFBaseModelOutput]=None, past_key_values=None, use_cache=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    labels=None, training=False, **kwargs):
    """
        Returns:

        Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, TFBartForConditionalGeneration
            import tensorflow as tf
            mname = 'facebook/bart-large'
            tokenizer = BartTokenizer.from_pretrained(mname)
            TXT = "My friends are <mask> but they eat too many carbs."
            model = TFBartForConditionalGeneration.from_pretrained(mname)
            batch = tokenizer([TXT], return_tensors='tf')
            logits = model(inputs=batch.input_ids).logits
            probs = tf.nn.softmax(logits[0])
            # probs[5] is associated with the mask token
        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask, encoder_outputs=
        encoder_outputs, past_key_values=past_key_values, use_cache=
        use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        labels=labels, training=training, kwargs_call=kwargs)
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.config.return_dict
    use_cache = inputs['use_cache'] if inputs['use_cache'
        ] is not None else self.config.use_cache
    if inputs['labels'] is not None:
        use_cache = False
        if inputs['decoder_input_ids'] is None:
            inputs['decoder_input_ids'] = self._shift_right(inputs['labels'])
    outputs = self.model(inputs['input_ids'], attention_mask=inputs[
        'attention_mask'], decoder_input_ids=inputs['decoder_input_ids'],
        encoder_outputs=inputs['encoder_outputs'], decoder_attention_mask=
        inputs['decoder_attention_mask'], past_key_values=inputs[
        'past_key_values'], use_cache=use_cache, output_attentions=inputs[
        'output_attentions'], output_hidden_states=inputs[
        'output_hidden_states'], return_dict=return_dict)
    lm_logits = self.model.shared(outputs[0], mode='linear')
    lm_logits = lm_logits + self.final_logits_bias
    masked_lm_loss = None if inputs['labels'] is None else self.compute_loss(
        inputs['labels'], lm_logits)
    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return (masked_lm_loss,
            ) + output if masked_lm_loss is not None else output
    return TFSeq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits,
        past_key_values=outputs.past_key_values, decoder_hidden_states=
        outputs.decoder_hidden_states, decoder_attentions=outputs.
        decoder_attentions, encoder_last_hidden_state=outputs.
        last_hidden_state, encoder_hidden_states=outputs.
        encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)
