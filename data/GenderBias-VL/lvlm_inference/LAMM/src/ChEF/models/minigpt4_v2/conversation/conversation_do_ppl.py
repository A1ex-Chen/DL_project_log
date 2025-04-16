def do_ppl(self, embs_list, answer_list, answer_options, calib=False):
    max_emb_token = max([x.shape[1] for x in embs_list])
    padding_len_list = [(max_emb_token - x.shape[1]) for x in embs_list]
    target_ids = []
    answer_start_indices = []
    answer_end_indices = []
    answer_token_list = []
    template_token_list = []
    for template, option in zip(answer_list, answer_options):
        template_token = self.model.llama_tokenizer(template,
            return_tensors='pt', add_special_tokens=False).input_ids
        template_token_list.append(template_token)
        option_token = self.model.llama_tokenizer(option, return_tensors=
            'pt', add_special_tokens=False).input_ids
        target_ids.append(template_token)
        token_len = len(option_token[0])
        for index in range(len(template_token[0])):
            if torch.all(template_token[0][index:index + token_len] ==
                option_token[0]):
                answer_start_indices.append(index)
                answer_end_indices.append(index + token_len)
                answer_token_list.append(option_token[0])
                break
        assert len(answer_start_indices) == len(template_token_list
            ), 'tokenizer encode answer in template different from answer only'
    target_ids = torch.cat([F.pad(x, (max_emb_token - x.shape[1], 0, 0, 0),
        value=-100) for x in target_ids], dim=0).to(self.device)
    embs_list = torch.cat([F.pad(x, (0, 0, max_emb_token - x.shape[1], 0, 0,
        0), value=0) for x in embs_list], dim=0)
    att_mask = torch.ones(embs_list.shape[:-1])
    for idx, padding_len in enumerate(padding_len_list):
        att_mask[idx, :padding_len] = 0
    att_mask = att_mask.bool().to(self.device)
    outputs = self.model.llama_model(inputs_embeds=embs_list,
        attention_mask=att_mask, return_dict=True)
    logits = outputs['logits'][:, :-1]
    target_ids = target_ids[:, 1:]
    loss_mask = target_ids != -100
    results = []
    if calib:
        for idx, item_logits in enumerate(logits):
            score = 0.0
            item_prob = F.softmax(item_logits[loss_mask[idx]][
                answer_start_indices[idx]:answer_end_indices[idx]], dim=-1)
            for jdx in range(answer_end_indices[idx] - answer_start_indices
                [idx]):
                score += torch.log(item_prob[jdx, answer_token_list[idx][jdx]]
                    ).item()
            score = score / len(answer_token_list[idx])
            results.append(score)
    else:
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1), ignore_index=-100, reduction='none')
        loss = loss.reshape(-1, target_ids.shape[1]).float()
        for idx, item_loss in enumerate(loss):
            results.append(item_loss[loss_mask[idx]][answer_start_indices[
                idx]:answer_end_indices[idx]].mean().item())
    return results
