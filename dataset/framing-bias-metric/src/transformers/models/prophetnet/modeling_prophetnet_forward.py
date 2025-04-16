@add_start_docstrings_to_model_forward(PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=ProphetNetDecoderLMOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
    output_attentions=None, output_hidden_states=None, return_dict=None):
    """
        encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last ``decoder_input_ids``
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import ProphetNetTokenizer, ProphetNetForCausalLM
            >>> import torch

            >>> tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = ProphetNetForCausalLM.from_pretrained('patrickvonplaten/prophetnet-decoder-clm-large-uncased')
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits

            >>> # Model can also be used with EncoderDecoder framework
            >>> from transformers import BertTokenizer, EncoderDecoderModel, ProphetNetTokenizer
            >>> import torch

            >>> tokenizer_enc = BertTokenizer.from_pretrained('bert-large-uncased')
            >>> tokenizer_dec = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-large-uncased", "patrickvonplaten/prophetnet-decoder-clm-large-uncased")

            >>> ARTICLE = (
            ... "the us state department said wednesday it had received no "
            ... "formal word from bolivia that it was expelling the us ambassador there "
            ... "but said the charges made against him are `` baseless ."
            ... )
            >>> input_ids = tokenizer_enc(ARTICLE, return_tensors="pt").input_ids
            >>> labels = tokenizer_dec("us rejects charges against its ambassador in bolivia", return_tensors="pt").input_ids
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-1], labels=labels[:, 1:])

            >>> loss = outputs.loss
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.decoder(input_ids=input_ids, attention_mask=
        attention_mask, encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask, past_key_values=
        past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    batch_size, sequence_length = (input_ids.shape if input_ids is not None
         else inputs_embeds.shape[:2])
    predicting_streams = outputs[1].view(batch_size, self.config.ngram,
        sequence_length, -1)
    predict_logits = self.lm_head(predicting_streams)
    logits = predict_logits[:, 0]
    logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None
    loss = None
    if labels is not None:
        loss = self._compute_loss(predict_logits, labels)
    if not return_dict:
        all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
        return (loss,) + all_logits + outputs[2:
            ] if loss is not None else all_logits + outputs[2:]
    else:
        return ProphetNetDecoderLMOutput(loss=loss, logits=logits,
            logits_ngram=logits_ngram, past_key_values=outputs.
            past_key_values, hidden_states=outputs.hidden_states,
            hidden_states_ngram=outputs.hidden_states_ngram, attentions=
            outputs.attentions, ngram_attentions=outputs.ngram_attentions,
            cross_attentions=outputs.cross_attentions)
