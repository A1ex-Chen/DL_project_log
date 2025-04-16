def compute_text_embeddings(prompt):
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt,
            tokenizer_max_length=args.tokenizer_max_length)
        prompt_embeds = encode_prompt(text_encoder, text_inputs.input_ids,
            text_inputs.attention_mask, text_encoder_use_attention_mask=
            args.text_encoder_use_attention_mask)
    return prompt_embeds
