def _encode_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance, negative_prompt=None, prompt_embeds:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, lora_scale: Optional[float]=None):
    deprecation_message = (
        '`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple.'
        )
    deprecate('_encode_prompt()', '1.0.0', deprecation_message,
        standard_warn=False)
    prompt_embeds_tuple = self.encode_prompt(prompt=prompt, device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds, lora_scale=lora_scale)
    prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
    return prompt_embeds
