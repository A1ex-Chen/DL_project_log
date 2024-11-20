def _rank_answers(self, samples, answer_list, num_ans_candidates):
    """
        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
    answer_candidates = self.tokenizer(answer_list, padding='longest',
        return_tensors='pt').to(self.device)
    answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id
    answer_ids = answer_candidates.input_ids
    answer_atts = answer_candidates.attention_mask
    question_output, _ = self.forward_encoder(samples)
    question_states = question_output.last_hidden_state
    tokenized_question = samples['tokenized_text']
    question_atts = tokenized_question.attention_mask
    num_ques = question_states.size(0)
    start_ids = answer_ids[0, 0].repeat(num_ques, 1)
    start_output = self.text_decoder(start_ids, encoder_hidden_states=
        question_states, encoder_attention_mask=question_atts, return_dict=
        True, reduction='none')
    logits = start_output.logits[:, 0, :]
    answer_first_token = answer_ids[:, 1]
    prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=
        answer_first_token)
    topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)
    input_ids = []
    input_atts = []
    for b, topk_id in enumerate(topk_ids):
        input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
        input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
    input_ids = torch.cat(input_ids, dim=0)
    input_atts = torch.cat(input_atts, dim=0)
    targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.
        pad_token_id, -100)
    question_states = tile(question_states, 0, num_ans_candidates)
    question_atts = tile(question_atts, 0, num_ans_candidates)
    output = self.text_decoder(input_ids, attention_mask=input_atts,
        encoder_hidden_states=question_states, encoder_attention_mask=
        question_atts, labels=targets_ids, return_dict=True, reduction='none')
    log_probs_sum = -output.loss
    log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)
    max_topk_ids = log_probs_sum.argmax(dim=1)
    max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
    answers = [answer_list[max_id] for max_id in max_ids]
    return answers
