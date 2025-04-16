def _encode_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance):
    len(prompt) if isinstance(prompt, list) else 1
    self.tokenizer.pad_token_id = 0
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding='longest',
        return_tensors='pt').input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
        ] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.
            tokenizer.model_max_length - 1:-1])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
    text_encoder_output = self.text_encoder(text_input_ids.to(device))
    prompt_embeds = text_encoder_output.text_embeds
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,
        dim=0)
    prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1,
        keepdim=True)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = math.sqrt(prompt_embeds.shape[1]) * prompt_embeds
    return prompt_embeds
