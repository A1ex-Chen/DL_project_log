def compute_embeddings(batch, proportion_empty_prompts, text_encoders,
    tokenizers, is_train=True):
    original_size = args.resolution, args.resolution
    target_size = args.resolution, args.resolution
    crops_coords_top_left = (args.crops_coords_top_left_h, args.
        crops_coords_top_left_w)
    prompt_batch = batch[args.caption_column]
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch,
        text_encoders, tokenizers, proportion_empty_prompts, is_train)
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    prompt_embeds = prompt_embeds.to(accelerator.device)
    add_text_embeds = add_text_embeds.to(accelerator.device)
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.
        dtype)
    unet_added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
        add_time_ids}
    return {'prompt_embeds': prompt_embeds, **unet_added_cond_kwargs}
