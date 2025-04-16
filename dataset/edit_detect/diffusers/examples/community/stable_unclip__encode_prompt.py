def _encode_prompt(self, prompt, device, num_images_per_prompt,
    do_classifier_free_guidance, text_model_output: Optional[Union[
    CLIPTextModelOutput, Tuple]]=None, text_attention_mask: Optional[torch.
    Tensor]=None):
    if text_model_output is None:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(prompt, padding='max_length',
            max_length=self.tokenizer.model_max_length, return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:,
                self.tokenizer.model_max_length:])
            logger.warning(
                f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
            text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length
                ]
        text_encoder_output = self.text_encoder(text_input_ids.to(device))
        text_embeddings = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state
    else:
        batch_size = text_model_output[0].shape[0]
        text_embeddings, text_encoder_hidden_states = text_model_output[0
            ], text_model_output[1]
        text_mask = text_attention_mask
    text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt,
        dim=0)
    text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(
        num_images_per_prompt, dim=0)
    text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        uncond_tokens = [''] * batch_size
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        uncond_text_mask = uncond_input.attention_mask.bool().to(device)
        uncond_embeddings_text_encoder_output = self.text_encoder(uncond_input
            .input_ids.to(device))
        uncond_embeddings = uncond_embeddings_text_encoder_output.text_embeds
        uncond_text_encoder_hidden_states = (
            uncond_embeddings_text_encoder_output.last_hidden_state)
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt)
        uncond_embeddings = uncond_embeddings.view(batch_size *
            num_images_per_prompt, seq_len)
        seq_len = uncond_text_encoder_hidden_states.shape[1]
        uncond_text_encoder_hidden_states = (uncond_text_encoder_hidden_states
            .repeat(1, num_images_per_prompt, 1))
        uncond_text_encoder_hidden_states = (uncond_text_encoder_hidden_states
            .view(batch_size * num_images_per_prompt, seq_len, -1))
        uncond_text_mask = uncond_text_mask.repeat_interleave(
            num_images_per_prompt, dim=0)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_encoder_hidden_states = torch.cat([
            uncond_text_encoder_hidden_states, text_encoder_hidden_states])
        text_mask = torch.cat([uncond_text_mask, text_mask])
    return text_embeddings, text_encoder_hidden_states, text_mask
