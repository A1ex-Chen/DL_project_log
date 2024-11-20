@add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'microsoft/deberta-base', output_type=SequenceClassifierOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, inputs_embeds=None, labels=None, output_attentions=
    None, output_hidden_states=None, return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.deberta(input_ids, token_type_ids=token_type_ids,
        attention_mask=attention_mask, position_ids=position_ids,
        inputs_embeds=inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
    encoder_layer = outputs[0]
    pooled_output = self.pooler(encoder_layer)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = None
    if labels is not None:
        if self.num_labels == 1:
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = loss_fn(logits, labels.view(-1))
        elif labels.dim() == 1 or labels.size(-1) == 1:
            label_index = (labels >= 0).nonzero()
            labels = labels.long()
            if label_index.size(0) > 0:
                labeled_logits = torch.gather(logits, 0, label_index.expand
                    (label_index.size(0), logits.size(1)))
                labels = torch.gather(labels, 0, label_index.view(-1))
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(labeled_logits.view(-1, self.num_labels).
                    float(), labels.view(-1))
            else:
                loss = torch.tensor(0).to(logits)
        else:
            log_softmax = torch.nn.LogSoftmax(-1)
            loss = -(log_softmax(logits) * labels).sum(-1).mean()
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    else:
        return SequenceClassifierOutput(loss=loss, logits=logits,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)
