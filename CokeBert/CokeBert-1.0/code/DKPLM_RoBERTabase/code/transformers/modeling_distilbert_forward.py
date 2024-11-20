def forward(self, input_ids=None, attention_mask=None, head_mask=None,
    inputs_embeds=None, labels=None):
    outputs = self.distilbert(input_ids, attention_mask=attention_mask,
        head_mask=head_mask, inputs_embeds=inputs_embeds)
    sequence_output = outputs[0]
    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)
    outputs = (logits,) + outputs[2:]
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs
    return outputs
