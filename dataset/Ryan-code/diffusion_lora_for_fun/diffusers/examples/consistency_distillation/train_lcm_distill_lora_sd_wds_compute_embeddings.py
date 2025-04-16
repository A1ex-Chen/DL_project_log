def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder,
    tokenizer, is_train=True):
    prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer,
        proportion_empty_prompts, is_train)
    return {'prompt_embeds': prompt_embeds}
