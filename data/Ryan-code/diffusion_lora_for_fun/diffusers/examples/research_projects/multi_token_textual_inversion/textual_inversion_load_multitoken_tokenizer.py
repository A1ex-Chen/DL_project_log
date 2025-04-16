def load_multitoken_tokenizer(tokenizer, text_encoder, learned_embeds_dict):
    for placeholder_token in learned_embeds_dict:
        placeholder_embeds = learned_embeds_dict[placeholder_token]
        num_vec_per_token = placeholder_embeds.shape[0]
        placeholder_embeds = placeholder_embeds.to(dtype=text_encoder.dtype)
        add_tokens(tokenizer, text_encoder, placeholder_token,
            num_vec_per_token=num_vec_per_token)
        placeholder_token_ids = tokenizer.encode(placeholder_token,
            add_special_tokens=False)
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = placeholder_embeds[i]
