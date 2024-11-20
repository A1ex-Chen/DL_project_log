def encode_prompt(batch, text_encoders, tokenizers,
    proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append('')
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(captions, padding='max_length',
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.
                device), output_hidden_states=True, return_dict=False)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {'prompt_embeds': prompt_embeds.cpu(), 'pooled_prompt_embeds':
        pooled_prompt_embeds.cpu()}
