def compute_text_embeddings(prompt, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
        prompt_embeds = prompt_embeds.to(accelerator.device)
    return prompt_embeds
