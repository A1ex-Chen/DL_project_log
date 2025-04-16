def encode_prompt(self, tokenizers, text_encoders, prompt: str,
    num_images_per_prompt: int=1, negative_prompt: str=None):
    device = text_encoders[0].device
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(prompt, padding='max_length', max_length=
            tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device),
            output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    if negative_prompt is None:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    else:
        negative_prompt = batch_size * [negative_prompt] if isinstance(
            negative_prompt, str) else negative_prompt
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(negative_prompt, padding='max_length',
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors='pt')
            negative_prompt_embeds = text_encoder(uncond_input.input_ids.to
                (device), output_hidden_states=True)
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            negative_prompt_embeds_list.append(negative_prompt_embeds)
        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list,
            dim=-1)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1,
        num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(batch_size *
        num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt
        ).view(bs_embed * num_images_per_prompt, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1,
        num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
    return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds)
