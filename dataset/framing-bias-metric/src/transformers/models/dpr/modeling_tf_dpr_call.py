@add_start_docstrings_to_model_forward(TF_DPR_READER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=TFDPRReaderOutput, config_class=
    _CONFIG_FOR_DOC)
def call(self, input_ids=None, attention_mask: Optional[tf.Tensor]=None,
    token_type_ids: Optional[tf.Tensor]=None, inputs_embeds: Optional[tf.
    Tensor]=None, output_attentions: bool=None, output_hidden_states: bool=
    None, return_dict=None, training: bool=False, **kwargs) ->Union[
    TFDPRReaderOutput, Tuple[tf.Tensor, ...]]:
    """
        Return:

        Examples::

            >>> from transformers import TFDPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = TFDPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', from_pt=True)
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='tf'
            ...     )
            >>> outputs = model(encoded_inputs)
            >>> start_logits = outputs.start_logits
            >>> end_logits = outputs.end_logits
            >>> relevance_logits = outputs.relevance_logits

        """
    inputs = input_processing(func=self.call, input_ids=input_ids,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        training=training, kwargs_call=kwargs)
    output_attentions = inputs['output_attentions'] if inputs[
        'output_attentions'] is not None else self.config.output_attentions
    output_hidden_states = inputs['output_hidden_states'] if inputs[
        'output_hidden_states'
        ] is not None else self.config.output_hidden_states
    return_dict = inputs['return_dict'] if inputs['return_dict'
        ] is not None else self.config.use_return_dict
    if inputs['input_ids'] is not None and inputs['inputs_embeds'] is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'
            )
    elif inputs['input_ids'] is not None:
        input_shape = shape_list(inputs['input_ids'])
    elif inputs['inputs_embeds'] is not None:
        input_shape = shape_list(inputs['inputs_embeds'])[:-1]
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')
    if inputs['attention_mask'] is None:
        inputs['attention_mask'] = tf.ones(input_shape, dtype=tf.dtypes.int32)
    if token_type_ids is None:
        token_type_ids = tf.zeros(input_shape, dtype=tf.dtypes.int32)
    return self.span_predictor(input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'], token_type_ids=inputs[
        'token_type_ids'], inputs_embeds=inputs['inputs_embeds'],
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict, training=inputs[
        'training'])
