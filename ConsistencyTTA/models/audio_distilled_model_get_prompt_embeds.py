def get_prompt_embeds(self, prompt, use_cf_guidance, num_samples_per_prompt=1):
    """ Return: 
            prompt_embeds of cond+uncond, prompt_mask_cf of cond+uncond, 
            prompt_embeds of cond only, prompt_mask of cond only
        """
    if use_cf_guidance:
        return self.encode_text_classifier_free(prompt, num_samples_per_prompt)
    else:
        prompt_embeds, prompt_mask = self.encode_text(prompt)
        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt,
            0)
        prompt_mask = prompt_mask.repeat_interleave(num_samples_per_prompt, 0)
    return prompt_embeds, prompt_mask, prompt_embeds, prompt_mask
