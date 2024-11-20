def do_generate(self, image_list: torch.Tensor, prompt: str, max_new_tokens,
    **kwargs):
    imgs = image_list.unsqueeze(0).to(self.device)
    prompts = [prompt]
    results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens,
        device=self.device)
    result = results[0].strip()
    return result
