def encode_prompts(text_encoders, tokenizers, prompts):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []
    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders,
            tokenizers, prompt)
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)
    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all
        )
