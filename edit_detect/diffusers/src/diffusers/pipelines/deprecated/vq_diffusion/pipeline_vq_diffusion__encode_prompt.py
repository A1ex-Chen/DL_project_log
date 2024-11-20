def _encode_prompt(self, prompt, num_images_per_prompt,
    do_classifier_free_guidance):
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
        removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.
            tokenizer.model_max_length:])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
        text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
    prompt_embeds = self.text_encoder(text_input_ids.to(self.device))[0]
    prompt_embeds = prompt_embeds / prompt_embeds.norm(dim=-1, keepdim=True)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,
        dim=0)
    if do_classifier_free_guidance:
        if self.learned_classifier_free_sampling_embeddings.learnable:
            negative_prompt_embeds = (self.
                learned_classifier_free_sampling_embeddings.embeddings)
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0
                ).repeat(batch_size, 1, 1)
        else:
            uncond_tokens = [''] * batch_size
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding=
                'max_length', max_length=max_length, truncation=True,
                return_tensors='pt')
            negative_prompt_embeds = self.text_encoder(uncond_input.
                input_ids.to(self.device))[0]
            negative_prompt_embeds = (negative_prompt_embeds /
                negative_prompt_embeds.norm(dim=-1, keepdim=True))
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len, -1)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds
