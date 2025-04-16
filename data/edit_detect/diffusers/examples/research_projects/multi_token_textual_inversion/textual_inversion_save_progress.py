def save_progress(tokenizer, text_encoder, accelerator, save_path):
    for placeholder_token in tokenizer.token_map:
        placeholder_token_ids = tokenizer.encode(placeholder_token,
            add_special_tokens=False)
        learned_embeds = accelerator.unwrap_model(text_encoder
            ).get_input_embeddings().weight[placeholder_token_ids]
        if len(placeholder_token_ids) == 1:
            learned_embeds = learned_embeds[None]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()
            }
        torch.save(learned_embeds_dict, save_path)
