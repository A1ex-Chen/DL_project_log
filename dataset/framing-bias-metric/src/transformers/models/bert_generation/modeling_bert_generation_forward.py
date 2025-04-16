@add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.
    format('batch_size, sequence_length'))
@replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, position_ids=None,
    head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
    encoder_attention_mask=None, labels=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    """
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
            >>> import torch

            >>> tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
            >>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            >>> config.is_decoder = True
            >>> model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=config)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.bert(input_ids, attention_mask=attention_mask,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    sequence_output = outputs[0]
    prediction_scores = self.lm_head(sequence_output)
    lm_loss = None
    if labels is not None:
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.
            vocab_size), labels.view(-1))
    if not return_dict:
        output = (prediction_scores,) + outputs[1:]
        return (lm_loss,) + output if lm_loss is not None else output
    return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=
        prediction_scores, hidden_states=outputs.hidden_states, attentions=
        outputs.attentions, cross_attentions=outputs.cross_attentions)
