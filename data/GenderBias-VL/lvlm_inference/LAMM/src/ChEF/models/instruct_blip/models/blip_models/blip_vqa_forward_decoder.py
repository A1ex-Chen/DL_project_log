def forward_decoder(self, samples, encoder_out, **kwargs):
    answers = self.tokenizer(samples['answer'], padding='longest',
        return_tensors='pt').to(self.device)
    answers.input_ids[:, 0] = self.tokenizer.bos_token_id
    answer_targets = answers.input_ids.masked_fill(answers.input_ids ==
        self.tokenizer.pad_token_id, -100)
    question_states = []
    question_atts = []
    question = samples['tokenized_text']
    question_output = encoder_out
    for b, n in enumerate(samples['n_answers']):
        question_states += [question_output.last_hidden_state[b]] * n
        question_atts += [question.attention_mask[b]] * n
    question_states = torch.stack(question_states, dim=0)
    question_atts = torch.stack(question_atts, dim=0)
    answer_output = self.text_decoder(answers.input_ids, attention_mask=
        answers.attention_mask, encoder_hidden_states=question_states,
        encoder_attention_mask=question_atts, labels=answer_targets,
        return_dict=True, reduction='none')
    loss = samples['weight'] * answer_output.loss
    bsz = samples['image'].size(0)
    loss = loss.sum() / bsz
    return loss, answer_output, answer_targets
