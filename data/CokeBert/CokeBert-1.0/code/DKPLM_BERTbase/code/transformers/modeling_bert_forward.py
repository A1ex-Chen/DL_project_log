def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, start_positions=
    None, end_positions=None, answer_masks=None):
    outputs = self.bert(input_ids, attention_mask=attention_mask,
        token_type_ids=token_type_ids, position_ids=position_ids, head_mask
        =head_mask, inputs_embeds=inputs_embeds)
    sequence_output = outputs[0]
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    outputs = (start_logits, end_logits) + outputs[2:]
    if start_positions is not None and end_positions is not None:
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
        start_losses = [(loss_fct(start_logits, _start_positions) *
            _span_mask) for _start_positions, _span_mask in zip(torch.
            unbind(start_positions, dim=1), torch.unbind(answer_masks, dim=1))]
        end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask) for
            _end_positions, _span_mask in zip(torch.unbind(end_positions,
            dim=1), torch.unbind(answer_masks, dim=1))]
        total_loss = sum(start_losses + end_losses)
        total_loss = torch.mean(total_loss) / 2
        outputs = (total_loss,) + outputs
    return outputs
