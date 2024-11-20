@add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@replace_return_docstrings(output_type=XLNetForQuestionAnsweringOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, mems=None, perm_mask
    =None, target_mapping=None, token_type_ids=None, input_mask=None,
    head_mask=None, inputs_embeds=None, start_positions=None, end_positions
    =None, is_impossible=None, cls_index=None, p_mask=None, use_mems=None,
    output_attentions=None, output_hidden_states=None, return_dict=None, **
    kwargs):
    """
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        is_impossible (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for position (index) of the classification token to use as input for computing plausibility of the
            answer.
        p_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
            masked. 0.0 mean token is not masked.

        Returns:

        Example::

            >>> from transformers import XLNetTokenizer, XLNetForQuestionAnswering
            >>> import torch

            >>> tokenizer =  XLNetTokenizer.from_pretrained('xlnet-base-cased')
            >>> model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> start_positions = torch.tensor([1])
            >>> end_positions = torch.tensor([3])
            >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

            >>> loss = outputs.loss
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    transformer_outputs = self.transformer(input_ids, attention_mask=
        attention_mask, mems=mems, perm_mask=perm_mask, target_mapping=
        target_mapping, token_type_ids=token_type_ids, input_mask=
        input_mask, head_mask=head_mask, inputs_embeds=inputs_embeds,
        use_mems=use_mems, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs)
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
        if not return_dict:
            return (total_loss,) + transformer_outputs[1:]
        else:
            return XLNetForQuestionAnsweringOutput(loss=total_loss, mems=
                transformer_outputs.mems, hidden_states=transformer_outputs
                .hidden_states, attentions=transformer_outputs.attentions)
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
        if not return_dict:
            outputs = (start_top_log_probs, start_top_index,
                end_top_log_probs, end_top_index, cls_logits)
            return outputs + transformer_outputs[1:]
        else:
            return XLNetForQuestionAnsweringOutput(start_top_log_probs=
                start_top_log_probs, start_top_index=start_top_index,
                end_top_log_probs=end_top_log_probs, end_top_index=
                end_top_index, cls_logits=cls_logits, mems=
                transformer_outputs.mems, hidden_states=transformer_outputs
                .hidden_states, attentions=transformer_outputs.attentions)
