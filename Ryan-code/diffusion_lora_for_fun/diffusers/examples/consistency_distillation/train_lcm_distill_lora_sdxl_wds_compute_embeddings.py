def compute_embeddings(prompt_batch, original_sizes, crop_coords,
    proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
    target_size = args.resolution, args.resolution
    original_sizes = list(map(list, zip(*original_sizes)))
    crops_coords_top_left = list(map(list, zip(*crop_coords)))
    original_sizes = torch.tensor(original_sizes, dtype=torch.long)
    crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=torch
        .long)
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch,
        text_encoders, tokenizers, proportion_empty_prompts, is_train)
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = list(target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
    add_time_ids = torch.cat([original_sizes, crops_coords_top_left,
        add_time_ids], dim=-1)
    add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.
        dtype)
    prompt_embeds = prompt_embeds.to(accelerator.device)
    add_text_embeds = add_text_embeds.to(accelerator.device)
    unet_added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
        add_time_ids}
    return {'prompt_embeds': prompt_embeds, **unet_added_cond_kwargs}
