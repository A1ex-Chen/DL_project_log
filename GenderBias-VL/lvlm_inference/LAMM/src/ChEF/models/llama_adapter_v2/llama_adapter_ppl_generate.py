@torch.inference_mode()
def ppl_generate(self, imgs, prompts, answers, device='cuda'):
    bsz = len(imgs)
    params = self.llama.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    assert len(imgs) == len(prompts)
    assert len(imgs) == len(answers)
    with torch.cuda.amp.autocast():
        visual_query = self.forward_visual(imgs)
    prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    answers = [self.tokenizer.encode('Response: ' + x, bos=False, eos=False
        )[2:] for x in answers]
    prompts = [(prompt + answer) for prompt, answer in zip(prompts, answers)]
    total_len = max([len(t) for t in prompts])
    input_ids = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device
        ).long()
    target_ids = torch.full((bsz, total_len), -100).to(device).long()
    for k, (t, a) in enumerate(zip(prompts, answers)):
        input_ids[k, :len(t)] = torch.tensor(t).to(device).long()
        target_ids[k, len(t) - len(a):len(t)] = torch.tensor(a).to(device
            ).long()
    min_prompt_size = min([len(t) for t in prompts])
    input_text_mask = input_ids != self.tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    res_logits = []
    for cur_pos in range(start_pos, total_len + 1):
        with torch.cuda.amp.autocast():
            logits = self.forward_inference(visual_query, input_ids[:,
                prev_pos:cur_pos], prev_pos, ppl=True)
        res_logits.append(logits)
        next_token = torch.argmax(logits[:, -1], dim=-1)
        next_token = next_token.reshape(-1)
        if cur_pos < total_len:
            next_token = torch.where(input_text_mask[:, cur_pos], input_ids
                [:, cur_pos], next_token)
            input_ids[:, cur_pos] = next_token
        prev_pos = cur_pos
    res_logits = torch.cat(res_logits, dim=1)
    res_logits = res_logits[:, :-1]
    return res_logits, target_ids[:, 1:]
