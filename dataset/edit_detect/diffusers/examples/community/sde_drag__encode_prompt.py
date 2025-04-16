def _encode_prompt(self, input_ids, attention_mask,
    text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(self.device)
    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(self.device)
    else:
        attention_mask = None
    prompt_embeds = self.text_encoder(text_input_ids, attention_mask=
        attention_mask)
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds
