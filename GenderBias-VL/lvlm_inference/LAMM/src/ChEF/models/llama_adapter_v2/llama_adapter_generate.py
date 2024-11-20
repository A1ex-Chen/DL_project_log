@torch.inference_mode()
def generate(self, imgs, prompts, max_gen_len: int=256, temperature: float=
    0.1, top_p: float=0.75, device='cuda'):
    bsz = len(imgs)
    params = self.llama.params
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    assert len(imgs) == len(prompts)
    with torch.cuda.amp.autocast():
        visual_query = self.forward_visual(imgs)
    if isinstance(prompts[0], str):
        prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in
            prompts]
    min_prompt_size = min([len(t) for t in prompts])
    max_prompt_size = max([len(t) for t in prompts])
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
    tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).to(device
        ).long()
    for k, t in enumerate(prompts):
        tokens[k, :len(t)] = torch.tensor(t).to(device).long()
    input_text_mask = tokens != self.tokenizer.pad_id
    start_pos = min_prompt_size
    prev_pos = 0
    for cur_pos in range(start_pos, total_len):
        with torch.cuda.amp.autocast():
            logits = self.forward_inference(visual_query, tokens[:,
                prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:,
            cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
            break
        prev_pos = cur_pos
    decoded = []
    for i, t in enumerate(tokens.tolist()):
        t = t[len(prompts[i]):len(prompts[i]) + max_gen_len]
        try:
            t = t[:t.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        decoded.append(self.tokenizer.decode(t))
    return decoded
