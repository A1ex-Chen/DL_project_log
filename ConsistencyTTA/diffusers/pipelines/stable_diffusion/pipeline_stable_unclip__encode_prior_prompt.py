def _encode_prior_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance, text_model_output: Optional[Union[
    CLIPTextModelOutput, Tuple]]=None, text_attention_mask: Optional[torch.
    Tensor]=None):
    if text_model_output is None:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.prior_tokenizer(prompt, padding='max_length',
            max_length=self.prior_tokenizer.model_max_length, truncation=
            True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)
        untruncated_ids = self.prior_tokenizer(prompt, padding='longest',
            return_tensors='pt').input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
            ] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.prior_tokenizer.batch_decode(untruncated_ids
                [:, self.prior_tokenizer.model_max_length - 1:-1])
            logger.warning(
                f'The following part of your input was truncated because CLIP can only handle sequences up to {self.prior_tokenizer.model_max_length} tokens: {removed_text}'
                )
            text_input_ids = text_input_ids[:, :self.prior_tokenizer.
                model_max_length]
        prior_text_encoder_output = self.prior_text_encoder(text_input_ids.
            to(device))
        prompt_embeds = prior_text_encoder_output.text_embeds
        prior_text_encoder_hidden_states = (prior_text_encoder_output.
            last_hidden_state)
    else:
        batch_size = text_model_output[0].shape[0]
        prompt_embeds, prior_text_encoder_hidden_states = text_model_output[0
            ], text_model_output[1]
        text_mask = text_attention_mask
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,
        dim=0)
    prior_text_encoder_hidden_states = (prior_text_encoder_hidden_states.
        repeat_interleave(num_images_per_prompt, dim=0))
    text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        uncond_tokens = [''] * batch_size
        uncond_input = self.prior_tokenizer(uncond_tokens, padding=
            'max_length', max_length=self.prior_tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        uncond_text_mask = uncond_input.attention_mask.bool().to(device)
        negative_prompt_embeds_prior_text_encoder_output = (self.
            prior_text_encoder(uncond_input.input_ids.to(device)))
        negative_prompt_embeds = (
            negative_prompt_embeds_prior_text_encoder_output.text_embeds)
        uncond_prior_text_encoder_hidden_states = (
            negative_prompt_embeds_prior_text_encoder_output.last_hidden_state)
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len)
        seq_len = uncond_prior_text_encoder_hidden_states.shape[1]
        uncond_prior_text_encoder_hidden_states = (
            uncond_prior_text_encoder_hidden_states.repeat(1,
            num_images_per_prompt, 1))
        uncond_prior_text_encoder_hidden_states = (
            uncond_prior_text_encoder_hidden_states.view(batch_size *
            num_images_per_prompt, seq_len, -1))
        uncond_text_mask = uncond_text_mask.repeat_interleave(
            num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prior_text_encoder_hidden_states = torch.cat([
            uncond_prior_text_encoder_hidden_states,
            prior_text_encoder_hidden_states])
        text_mask = torch.cat([uncond_text_mask, text_mask])
    return prompt_embeds, prior_text_encoder_hidden_states, text_mask
