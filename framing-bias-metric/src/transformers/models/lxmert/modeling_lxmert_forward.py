@add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'unc-nlp/lxmert-base-uncased', output_type=
    LxmertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, visual_feats=None, visual_pos=None,
    attention_mask=None, visual_attention_mask=None, token_type_ids=None,
    inputs_embeds=None, labels=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    """
        labels: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            A one-hot representation of the correct answer

        Returns:
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    lxmert_output = self.lxmert(input_ids=input_ids, visual_feats=
        visual_feats, visual_pos=visual_pos, token_type_ids=token_type_ids,
        attention_mask=attention_mask, visual_attention_mask=
        visual_attention_mask, inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states, output_attentions=
        output_attentions, return_dict=return_dict)
    pooled_output = lxmert_output[2]
    answer_score = self.answer_head(pooled_output)
    loss = None
    if labels is not None:
        loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.
            view(-1))
    if not return_dict:
        output = (answer_score,) + lxmert_output[3:]
        return (loss,) + output if loss is not None else output
    return LxmertForQuestionAnsweringOutput(loss=loss,
        question_answering_score=answer_score, language_hidden_states=
        lxmert_output.language_hidden_states, vision_hidden_states=
        lxmert_output.vision_hidden_states, language_attentions=
        lxmert_output.language_attentions, vision_attentions=lxmert_output.
        vision_attentions, cross_encoder_attentions=lxmert_output.
        cross_encoder_attentions)
