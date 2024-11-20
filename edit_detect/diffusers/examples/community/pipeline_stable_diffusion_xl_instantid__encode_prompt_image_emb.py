def _encode_prompt_image_emb(self, prompt_image_emb, device, dtype,
    do_classifier_free_guidance):
    if isinstance(prompt_image_emb, torch.Tensor):
        prompt_image_emb = prompt_image_emb.clone().detach()
    else:
        prompt_image_emb = torch.tensor(prompt_image_emb)
    prompt_image_emb = prompt_image_emb.to(device=device, dtype=dtype)
    prompt_image_emb = prompt_image_emb.reshape([1, -1, self.
        image_proj_model_in_features])
    if do_classifier_free_guidance:
        prompt_image_emb = torch.cat([torch.zeros_like(prompt_image_emb),
            prompt_image_emb], dim=0)
    else:
        prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
    prompt_image_emb = self.image_proj_model(prompt_image_emb)
    return prompt_image_emb
