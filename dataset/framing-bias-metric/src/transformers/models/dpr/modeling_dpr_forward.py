@add_start_docstrings_to_model_forward(DPR_READER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=DPRReaderOutput, config_class=
    _CONFIG_FOR_DOC)
def forward(self, input_ids: Optional[Tensor]=None, attention_mask:
    Optional[Tensor]=None, inputs_embeds: Optional[Tensor]=None,
    output_attentions: bool=None, output_hidden_states: bool=None,
    return_dict=None) ->Union[DPRReaderOutput, Tuple[Tensor, ...]]:
    """
        Return:

        Examples::

            >>> from transformers import DPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='pt'
            ...     )
            >>> outputs = model(**encoded_inputs)
            >>> start_logits = outputs.stat_logits
            >>> end_logits = outputs.end_logits
            >>> relevance_logits = outputs.relevance_logits

        """
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'
            )
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')
    device = (input_ids.device if input_ids is not None else inputs_embeds.
        device)
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    return self.span_predictor(input_ids, attention_mask, inputs_embeds=
        inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
