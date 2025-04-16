def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(prompt, padding='max_length', max_length=
            tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding='longest',
            return_tensors='pt').input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
            ] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 
                tokenizer.model_max_length - 1:-1])
            logger.warning(
                f'The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}'
                )
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
