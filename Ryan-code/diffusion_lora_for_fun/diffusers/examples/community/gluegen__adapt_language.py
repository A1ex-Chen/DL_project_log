def _adapt_language(self, prompt_embeds: torch.Tensor):
    prompt_embeds = prompt_embeds / 3
    prompt_embeds = self.language_adapter(prompt_embeds) * (self.
        tensor_norm / 2)
    return prompt_embeds
