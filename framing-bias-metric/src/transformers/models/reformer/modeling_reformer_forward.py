@add_start_docstrings_to_model_forward(REFORMER_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'google/reformer-crime-and-punishment', output_type=
    QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, position_ids=None, attention_mask=None,
    head_mask=None, inputs_embeds=None, num_hashes=None, start_positions=
    None, end_positions=None, output_hidden_states=None, output_attentions=
    None, return_dict=None):
    """
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    reformer_outputs = self.reformer(input_ids, position_ids=position_ids,
        attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=
        inputs_embeds, num_hashes=num_hashes, use_cache=False,
        output_hidden_states=output_hidden_states, output_attentions=
        output_attentions, return_dict=return_dict)
    sequence_output = reformer_outputs[0]
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    total_loss = None
    if start_positions is not None and end_positions is not None:
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
    if not return_dict:
        output = (start_logits, end_logits) + reformer_outputs[1:]
        return (total_loss,) + output if total_loss is not None else output
    return QuestionAnsweringModelOutput(loss=total_loss, start_logits=
        start_logits, end_logits=end_logits, hidden_states=reformer_outputs
        .hidden_states, attentions=reformer_outputs.attentions)
