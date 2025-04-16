def save_new_embed(text_encoder, modifier_token_id, accelerator, args,
    output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info('Saving embeddings')
    learned_embeds = accelerator.unwrap_model(text_encoder
        ).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f'{output_dir}/{y}.bin'
        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename,
                metadata={'format': 'pt'})
        else:
            torch.save(learned_embeds_dict, filename)
