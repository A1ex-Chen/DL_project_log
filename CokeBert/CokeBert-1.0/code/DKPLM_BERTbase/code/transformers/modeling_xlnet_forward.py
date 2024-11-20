def forward(self, input_ids=None, attention_mask=None, mems=None, perm_mask
    =None, target_mapping=None, token_type_ids=None, input_mask=None,
    head_mask=None, inputs_embeds=None, start_positions=None, end_positions
    =None, is_impossible=None, cls_index=None, p_mask=None):
    transformer_outputs = self.transformer(input_ids, attention_mask=
        attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=
        target_mapping, token_type_ids=token_type_ids, input_mask=
        input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds)
    hidden_states = transformer_outputs[0]
    start_logits = self.start_logits(hidden_states, p_mask=p_mask)
    outputs = transformer_outputs[1:]
    if start_positions is not None and end_positions is not None:
        for x in (start_positions, end_positions, cls_index, is_impossible):
            if x is not None and x.dim() > 1:
                x.squeeze_(-1)
        end_logits = self.end_logits(hidden_states, start_positions=
            start_positions, p_mask=p_mask)
        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        if cls_index is not None and is_impossible is not None:
            cls_logits = self.answer_class(hidden_states, start_positions=
                start_positions, cls_index=cls_index)
            loss_fct_cls = nn.BCEWithLogitsLoss()
            cls_loss = loss_fct_cls(cls_logits, is_impossible)
            total_loss += cls_loss * 0.5
        outputs = (total_loss,) + outputs
    else:
        bsz, slen, hsz = hidden_states.size()
        start_log_probs = F.softmax(start_logits, dim=-1)
        start_top_log_probs, start_top_index = torch.topk(start_log_probs,
            self.start_n_top, dim=-1)
        start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
        start_states = torch.gather(hidden_states, -2, start_top_index_exp)
        start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
        hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
            start_states)
        p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
        end_logits = self.end_logits(hidden_states_expanded, start_states=
            start_states, p_mask=p_mask)
        end_log_probs = F.softmax(end_logits, dim=1)
        end_top_log_probs, end_top_index = torch.topk(end_log_probs, self.
            end_n_top, dim=1)
        end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top *
            self.end_n_top)
        end_top_index = end_top_index.view(-1, self.start_n_top * self.
            end_n_top)
        start_states = torch.einsum('blh,bl->bh', hidden_states,
            start_log_probs)
        cls_logits = self.answer_class(hidden_states, start_states=
            start_states, cls_index=cls_index)
        outputs = (start_top_log_probs, start_top_index, end_top_log_probs,
            end_top_index, cls_logits) + outputs
    return outputs
