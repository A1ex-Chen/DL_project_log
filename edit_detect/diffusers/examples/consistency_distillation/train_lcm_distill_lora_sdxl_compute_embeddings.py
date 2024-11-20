def compute_embeddings(prompt_batch, original_sizes, crop_coords,
    text_encoders, tokenizers, is_train=True):

    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = args.resolution, args.resolution
        add_time_ids = list(original_size + crops_coords_top_left + target_size
            )
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch,
        text_encoders, tokenizers, is_train)
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = torch.cat([compute_time_ids(s, c) for s, c in zip(
        original_sizes, crop_coords)])
    prompt_embeds = prompt_embeds.to(accelerator.device)
    add_text_embeds = add_text_embeds.to(accelerator.device)
    unet_added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
        add_time_ids}
    return {'prompt_embeds': prompt_embeds, **unet_added_cond_kwargs}
