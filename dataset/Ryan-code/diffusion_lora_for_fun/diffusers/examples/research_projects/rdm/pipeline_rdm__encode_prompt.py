def _encode_prompt(self, prompt):
    text_inputs = self.tokenizer(prompt, padding='max_length', max_length=
        self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    text_input_ids = text_inputs.input_ids
    if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
        removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.
            tokenizer.model_max_length:])
        logger.warning(
            f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
            )
        text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]
    prompt_embeds = self.clip.get_text_features(text_input_ids.to(self.device))
    prompt_embeds = prompt_embeds / torch.linalg.norm(prompt_embeds, dim=-1,
        keepdim=True)
    prompt_embeds = prompt_embeds[:, None, :]
    return prompt_embeds
