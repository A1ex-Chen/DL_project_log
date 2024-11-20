def _encode_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance):
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    text_mask = text_inputs.attention_mask.bool().to(device)
    text_encoder_output = self.text_encoder(text_input_ids.to(device))
    prompt_embeds = text_encoder_output.text_embeds
    text_encoder_hidden_states = text_encoder_output.last_hidden_state
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt,
        dim=0)
    text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(
        num_images_per_prompt, dim=0)
    text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        uncond_tokens = [''] * batch_size
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt')
        uncond_text_mask = uncond_input.attention_mask.bool().to(device)
        negative_prompt_embeds_text_encoder_output = self.text_encoder(
            uncond_input.input_ids.to(device))
        negative_prompt_embeds = (negative_prompt_embeds_text_encoder_output
            .text_embeds)
        uncond_text_encoder_hidden_states = (
            negative_prompt_embeds_text_encoder_output.last_hidden_state)
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1,
            num_images_per_prompt)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
            num_images_per_prompt, seq_len)
        seq_len = uncond_text_encoder_hidden_states.shape[1]
        uncond_text_encoder_hidden_states = (uncond_text_encoder_hidden_states
            .repeat(1, num_images_per_prompt, 1))
        uncond_text_encoder_hidden_states = (uncond_text_encoder_hidden_states
            .view(batch_size * num_images_per_prompt, seq_len, -1))
        uncond_text_mask = uncond_text_mask.repeat_interleave(
            num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        text_encoder_hidden_states = torch.cat([
            uncond_text_encoder_hidden_states, text_encoder_hidden_states])
        text_mask = torch.cat([uncond_text_mask, text_mask])
    return prompt_embeds, text_encoder_hidden_states, text_mask
