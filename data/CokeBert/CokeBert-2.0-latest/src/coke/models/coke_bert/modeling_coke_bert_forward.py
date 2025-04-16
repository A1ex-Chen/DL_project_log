def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    input_ent=None, ent_mask=None, k_1=None, v_1=None, k_2=None, v_2=None):
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.cokebert(input_ids, attention_mask=attention_mask,
        token_type_ids=token_type_ids, position_ids=position_ids, head_mask
        =head_mask, inputs_embeds=inputs_embeds, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict, input_ent=input_ent.long(), ent_mask=
        ent_mask, k_1=k_1, v_1=v_1, k_2=k_2, v_2=v_2)
    seq_output = outputs[0]
    head = seq_output[input_ids == 1601]
    tail = seq_output[input_ids == 1089]
    pooled_output = torch.cat([head, tail], -1)
    pooled_output = self.dense(pooled_output)
    pooled_output = self.activation(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = 'regression'
            elif self.num_labels > 1 and (labels.dtype == torch.long or 
                labels.dtype == torch.int):
                self.config.problem_type = 'single_label_classification'
            else:
                self.config.problem_type = 'multi_label_classification'
        if self.config.problem_type == 'regression':
            loss_fct = nn.MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == 'single_label_classification':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == 'multi_label_classification':
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    if not return_dict:
        output = (logits,) + outputs[2:]
        return (loss,) + output if loss is not None else output
    return SequenceClassifierOutput(loss=loss, logits=logits)
