def forward(self, input_modal, input_ids=None, modal_start_tokens=None,
    modal_end_tokens=None, attention_mask=None, token_type_ids=None,
    modal_token_type_ids=None, position_ids=None, modal_position_ids=None,
    head_mask=None, inputs_embeds=None, labels=None, return_dict=None):
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.mmbt(input_modal=input_modal, input_ids=input_ids,
        modal_start_tokens=modal_start_tokens, modal_end_tokens=
        modal_end_tokens, attention_mask=attention_mask, token_type_ids=
        token_type_ids, modal_token_type_ids=modal_token_type_ids,
        position_ids=position_ids, modal_position_ids=modal_position_ids,
        head_mask=head_mask, inputs_embeds=inputs_embeds, return_dict=
        return_dict)
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = None
    if labels is not None:
        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    if not return_dict:
        output = (logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states
        =outputs.hidden_states, attentions=outputs.attentions)
