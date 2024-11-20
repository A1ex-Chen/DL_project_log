def save_progress(text_encoder, placeholder_token_ids, accelerator, args,
    save_path, safe_serialization=True):
    logger.info('Saving embeddings')
    learned_embeds = accelerator.unwrap_model(text_encoder
        ).get_input_embeddings().weight[min(placeholder_token_ids):max(
        placeholder_token_ids) + 1]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().
        cpu()}
    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path,
            metadata={'format': 'pt'})
    else:
        torch.save(learned_embeds_dict, save_path)
