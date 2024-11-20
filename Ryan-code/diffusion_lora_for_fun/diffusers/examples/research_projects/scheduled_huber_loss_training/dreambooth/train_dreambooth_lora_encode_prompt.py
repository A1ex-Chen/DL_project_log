def encode_prompt(text_encoder, input_ids, attention_mask,
    text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None
    prompt_embeds = text_encoder(text_input_ids, attention_mask=
        attention_mask, return_dict=False)
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds
