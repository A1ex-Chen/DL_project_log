def encode_text_classifier_free(self, prompt, num_samples_per_prompt):
    cond_prompt_embeds, cond_prompt_mask = self.encode_text(prompt)
    cond_prompt_embeds = cond_prompt_embeds.repeat_interleave(
        num_samples_per_prompt, 0)
    cond_prompt_mask = cond_prompt_mask.repeat_interleave(
        num_samples_per_prompt, 0)
    uncond_tokens = [''] * len(prompt)
    negative_prompt_embeds, uncond_prompt_mask = self.encode_text(uncond_tokens
        , max_length=cond_prompt_embeds.shape[1], padding='max_length')
    negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
        num_samples_per_prompt, 0)
    uncond_prompt_mask = uncond_prompt_mask.repeat_interleave(
        num_samples_per_prompt, 0)
    """ For classifier free guidance, we need to do two forward passes.
            We concatenate the unconditional and text embeddings into a single batch 
            to avoid doing two forward passes
        """
    prompt_embeds = torch.cat([negative_prompt_embeds, cond_prompt_embeds])
    prompt_mask = torch.cat([uncond_prompt_mask, cond_prompt_mask])
    return prompt_embeds, prompt_mask, cond_prompt_embeds, cond_prompt_mask
