def save_progress(text_encoder, placeholder_token_ids, accelerator, args,
    save_path):
    logger.info('Saving embeddings')
    learned_embeds = accelerator.unwrap_model(text_encoder
        ).get_input_embeddings().weight[min(placeholder_token_ids):max(
        placeholder_token_ids) + 1]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().
        cpu()}
    torch.save(learned_embeds_dict, save_path)
