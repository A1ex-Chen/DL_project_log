def _generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
    encoder_out, _ = self.forward_encoder(samples)
    question_output = encoder_out
    question_states = question_output.last_hidden_state.repeat_interleave(
        num_beams, dim=0)
    question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long
        ).to(self.device)
    model_kwargs = {'encoder_hidden_states': question_states,
        'encoder_attention_mask': question_atts}
    bsz = samples['image'].size(0)
    bos_ids = torch.full((bsz, 1), fill_value=self.tokenizer.bos_token_id,
        device=self.device)
    outputs = self.text_decoder.generate(input_ids=bos_ids, max_length=
        max_length, min_length=min_length, num_beams=num_beams,
        eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.
        tokenizer.pad_token_id, **model_kwargs)
    answers = []
    for output in outputs:
        answer = self.tokenizer.decode(output, skip_special_tokens=True)
        answers.append(answer)
    return answers
