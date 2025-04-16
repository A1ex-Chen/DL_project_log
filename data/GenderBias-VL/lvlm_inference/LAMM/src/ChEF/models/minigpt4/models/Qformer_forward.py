def forward(self, input_ids=None, attention_mask=None, position_ids=None,
    head_mask=None, query_embeds=None, encoder_hidden_states=None,
    encoder_attention_mask=None, labels=None, output_attentions=None,
    output_hidden_states=None, return_dict=None, return_logits=False,
    is_decoder=False):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.bert(input_ids, attention_mask=attention_mask,
        position_ids=position_ids, head_mask=head_mask, query_embeds=
        query_embeds, encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, is_decoder=is_decoder)
    if query_embeds is not None:
        sequence_output = outputs[0][:, query_embeds.shape[1]:, :]
    prediction_scores = self.cls(sequence_output)
    if return_logits:
        return prediction_scores
    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.
            vocab_size), labels.view(-1))
    if not return_dict:
        output = (prediction_scores,) + outputs[2:]
        return (masked_lm_loss,
            ) + output if masked_lm_loss is not None else output
    return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
