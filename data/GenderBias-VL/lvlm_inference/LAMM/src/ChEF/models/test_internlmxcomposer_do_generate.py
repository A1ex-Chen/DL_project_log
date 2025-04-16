def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
    image, _, _ = self.model.img2emb(image_list)
    inputs, im_mask = self.model.interleav_wrap_chat(self.tokenizer, prompt,
        image)
    inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.
        is_tensor(v)}
    eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.
        convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]]
    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0, top_p=0.8, use_cache=True,
        eos_token_id=eos_token_id, repetition_penalty=1.005, im_mask=
        im_mask, **kwargs)
    outputs = outputs[0].cpu().tolist()
    response = self.tokenizer.decode(outputs, skip_special_tokens=True)
    response = response.split('[UNUSED_TOKEN_145]')[0]
    return response
