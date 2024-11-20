def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
            output_hidden_states=True)
    return prompt_embeds[0]
