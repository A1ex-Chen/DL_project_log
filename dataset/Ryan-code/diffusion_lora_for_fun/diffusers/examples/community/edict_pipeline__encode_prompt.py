def _encode_prompt(self, prompt: str, negative_prompt: Optional[str]=None,
    do_classifier_free_guidance: bool=False):
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    prompt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device)
        ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=
        self.device)
    if do_classifier_free_guidance:
        uncond_tokens = '' if negative_prompt is None else negative_prompt
        uncond_input = self.tokenizer(uncond_tokens, padding='max_length',
            max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors='pt')
        negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.
            to(self.device)).last_hidden_state
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds
