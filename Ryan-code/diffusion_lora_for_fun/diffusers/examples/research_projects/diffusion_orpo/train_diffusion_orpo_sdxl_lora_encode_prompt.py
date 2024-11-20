@torch.no_grad()
def encode_prompt(text_encoders, text_input_ids_list):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = text_input_ids_list[i]
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
            output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
