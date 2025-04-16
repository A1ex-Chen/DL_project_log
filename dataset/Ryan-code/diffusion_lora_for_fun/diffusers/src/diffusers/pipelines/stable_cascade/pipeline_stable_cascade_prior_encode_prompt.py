def encode_prompt(self, device, batch_size, num_images_per_prompt,
    do_classifier_free_guidance, prompt=None, negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor]=None, prompt_embeds_pooled:
    Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.
    Tensor]=None, negative_prompt_embeds_pooled: Optional[torch.Tensor]=None):
    if prompt_embeds is None:
        text_inputs = self.tokenizer(prompt, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding='longest',
            return_tensors='pt').input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
            ] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, 
                self.tokenizer.model_max_length - 1:-1])
            logger.warning(
                f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length
                ]
            attention_mask = attention_mask[:, :self.tokenizer.model_max_length
                ]
        text_encoder_output = self.text_encoder(text_input_ids.to(device),
            attention_mask=attention_mask.to(device), output_hidden_states=True
            )
        prompt_embeds = text_encoder_output.hidden_states[-1]
        if prompt_embeds_pooled is None:
            prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=
        device)
    prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=self.text_encoder.
        dtype, device=device)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,
        dim=0)
    prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(
        num_images_per_prompt, dim=0)
    if negative_prompt_embeds is None and do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [''] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
        else:
            uncond_tokens = negative_prompt
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        negative_prompt_embeds_text_encoder_output = self.text_encoder(
            uncond_input.input_ids.to(device), attention_mask=uncond_input.
            attention_mask.to(device), output_hidden_states=True)
        negative_prompt_embeds = (negative_prompt_embeds_text_encoder_output
            .hidden_states[-1])
        negative_prompt_embeds_pooled = (
            negative_prompt_embeds_text_encoder_output.text_embeds.unsqueeze(1)
            )
    if do_classifier_free_guidance:
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.
            text_encoder.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len, -1)
        seq_len = negative_prompt_embeds_pooled.shape[1]
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(dtype
            =self.text_encoder.dtype, device=device)
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.repeat(
            1, num_images_per_prompt, 1)
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
            batch_size * num_images_per_prompt, seq_len, -1)
    return (prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds,
        negative_prompt_embeds_pooled)
