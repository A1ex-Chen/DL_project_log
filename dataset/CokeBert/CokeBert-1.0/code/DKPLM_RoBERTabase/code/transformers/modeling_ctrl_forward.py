def forward(self, input_ids=None, past=None, attention_mask=None,
    token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=
    None, labels=None):
    transformer_outputs = self.transformer(input_ids, past=past,
        attention_mask=attention_mask, token_type_ids=token_type_ids,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds)
    hidden_states = transformer_outputs[0]
    lm_logits = self.lm_head(hidden_states)
    outputs = (lm_logits,) + transformer_outputs[1:]
    if labels is not None:
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        outputs = (loss,) + outputs
    return outputs
