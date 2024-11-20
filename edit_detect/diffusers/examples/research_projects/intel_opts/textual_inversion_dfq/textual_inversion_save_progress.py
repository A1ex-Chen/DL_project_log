def save_progress(text_encoder, placeholder_token_id, accelerator, args,
    save_path):
    logger.info('Saving embeddings')
    learned_embeds = accelerator.unwrap_model(text_encoder
        ).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().
        cpu()}
    torch.save(learned_embeds_dict, save_path)
