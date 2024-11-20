def tokenize_conversation(self, conv_q, conv_a):
    """concatenate conversation and make sure the model is only trained to regress the answer"""
    to_regress_token_ids_list = []
    targets_list = []
    batch_size = len(conv_q)
    for batch_idx in range(batch_size):
        questions, answers = conv_q[batch_idx], conv_a[batch_idx]
        questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token +
            q, return_tensors='pt', add_special_tokens=False).to(self.
            device) for q in questions[1:]]
        answers = [self.llama_tokenizer(a + self.end_sym, return_tensors=
            'pt', add_special_tokens=False).to(self.device) for a in answers]
        cur_id = []
        cur_target = []
        for i in range(len(questions)):
            cur_id.append(answers[i].input_ids)
            cur_target.append(answers[i].input_ids)
            cur_id.append(questions[i].input_ids)
            cur_target.append(torch.ones_like(questions[i].input_ids) * -100)
        cur_id.append(answers[-1].input_ids)
        cur_target.append(answers[-1].input_ids)
        cur_id = torch.cat(cur_id, dim=1)
        cur_target = torch.cat(cur_target, dim=1)
        to_regress_token_ids_list.append(cur_id)
        targets_list.append(cur_target)
    max_len = min(max([target.shape[1] for target in targets_list]), self.
        max_txt_len)
    to_regress_token_ids = torch.ones([batch_size, max_len], dtype=cur_id.
        dtype, device=self.device) * self.llama_tokenizer.pad_token_id
    targets = torch.ones([batch_size, max_len], dtype=cur_id.dtype, device=
        self.device) * -100
    for batch_idx in range(batch_size):
        cur_len = to_regress_token_ids_list[batch_idx].shape[1]
        to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[
            batch_idx][0, :max_len]
        targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]
    to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.
        pad_token_id).to(torch.int)
    return to_regress_token_ids, to_regress_token_attn, targets
