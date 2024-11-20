@torch.no_grad()
def do_generate(self, images, prompt, max_new_tokens=30, **kwargs):
    input_ids = tokenizer_image_token(prompt, self.tokenizer,
        IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
    stopping_criteria = KeywordsStoppingCriteria(self.stop_str, self.
        tokenizer, input_ids)
    input_token_len = input_ids.shape[1]
    with torch.inference_mode():
        output_ids = self.model.generate(input_ids, images=[images],
            do_sample=False, temperature=0, max_new_tokens=max_new_tokens,
            use_cache=True, stopping_criteria=[stopping_criteria])
    output = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
        skip_special_token=True)[0]
    output = output.strip()
    for idx in range(len(self.stop_str[0])):
        if output.endswith(self.stop_str[0][:idx + 1]):
            output = output[:-(idx + 1)]
            break
    output = output.strip()
    return output
