def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(
            text_encoders, tokenizers, prompts)
        add_text_embeds_all = pooled_prompt_embeds_all
        prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
        add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
    return prompt_embeds_all, add_text_embeds_all
