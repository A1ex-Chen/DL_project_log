def forward(self, input_ids=None, attention_mask=None, position_ids=None,
    head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
    encoder_attention_mask=None, labels=None, past_key_values=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, return_logits=False, is_decoder=True, reduction=
    'mean', mode='multimodal', soft_labels=None, alpha=0):
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
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if labels is not None:
        use_cache = False
    outputs = self.bert(input_ids, attention_mask=attention_mask,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask, past_key_values=
        past_key_values, use_cache=use_cache, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, is_decoder=is_decoder, mode=mode)
    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)
    if return_logits:
        return prediction_scores[:, :-1, :].contiguous()
    lm_loss = None
    if labels is not None:
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
        lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.
            vocab_size), labels.view(-1))
        if reduction == 'none':
            lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)
    if soft_labels is not None:
        loss_distill = -torch.sum(F.log_softmax(shifted_prediction_scores,
            dim=-1) * soft_labels, dim=-1)
        loss_distill = (loss_distill * (labels != -100)).sum(1)
        lm_loss = (1 - alpha) * lm_loss + alpha * loss_distill
    if not return_dict:
        output = (prediction_scores,) + outputs[2:]
        return (lm_loss,) + output if lm_loss is not None else output
    return CausalLMOutputWithCrossAttentions(loss=lm_loss, logits=
        prediction_scores, past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        cross_attentions=outputs.cross_attentions)
