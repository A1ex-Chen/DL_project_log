def get_weighted_text_embeddings_sdxl(pipe: StableDiffusionXLPipeline,
    prompt: str='', prompt_2: str=None, neg_prompt: str='', neg_prompt_2:
    str=None, num_images_per_prompt: int=1, device: Optional[torch.device]=
    None, clip_skip: Optional[int]=None):
    """
    This function can process long prompt with weights, no length limitation
    for Stable Diffusion XL

    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        prompt_2 (str)
        neg_prompt (str)
        neg_prompt_2 (str)
        num_images_per_prompt (int)
        device (torch.device)
        clip_skip (int)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    """
    device = device or pipe._execution_device
    if prompt_2:
        prompt = f'{prompt} {prompt_2}'
    if neg_prompt_2:
        neg_prompt = f'{neg_prompt} {neg_prompt_2}'
    prompt_t1 = prompt_t2 = prompt
    neg_prompt_t1 = neg_prompt_t2 = neg_prompt
    if isinstance(pipe, TextualInversionLoaderMixin):
        prompt_t1 = pipe.maybe_convert_prompt(prompt_t1, pipe.tokenizer)
        neg_prompt_t1 = pipe.maybe_convert_prompt(neg_prompt_t1, pipe.tokenizer
            )
        prompt_t2 = pipe.maybe_convert_prompt(prompt_t2, pipe.tokenizer_2)
        neg_prompt_t2 = pipe.maybe_convert_prompt(neg_prompt_t2, pipe.
            tokenizer_2)
    eos = pipe.tokenizer.eos_token_id
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(pipe.
        tokenizer, prompt_t1)
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt_t1)
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(pipe
        .tokenizer_2, prompt_t2)
    neg_prompt_tokens_2, neg_prompt_weights_2 = (
        get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt_t2))
    prompt_token_len = len(prompt_tokens)
    neg_prompt_token_len = len(neg_prompt_tokens)
    if prompt_token_len > neg_prompt_token_len:
        neg_prompt_tokens = neg_prompt_tokens + [eos] * abs(
            prompt_token_len - neg_prompt_token_len)
        neg_prompt_weights = neg_prompt_weights + [1.0] * abs(
            prompt_token_len - neg_prompt_token_len)
    else:
        prompt_tokens = prompt_tokens + [eos] * abs(prompt_token_len -
            neg_prompt_token_len)
        prompt_weights = prompt_weights + [1.0] * abs(prompt_token_len -
            neg_prompt_token_len)
    prompt_token_len_2 = len(prompt_tokens_2)
    neg_prompt_token_len_2 = len(neg_prompt_tokens_2)
    if prompt_token_len_2 > neg_prompt_token_len_2:
        neg_prompt_tokens_2 = neg_prompt_tokens_2 + [eos] * abs(
            prompt_token_len_2 - neg_prompt_token_len_2)
        neg_prompt_weights_2 = neg_prompt_weights_2 + [1.0] * abs(
            prompt_token_len_2 - neg_prompt_token_len_2)
    else:
        prompt_tokens_2 = prompt_tokens_2 + [eos] * abs(prompt_token_len_2 -
            neg_prompt_token_len_2)
        prompt_weights_2 = prompt_weights + [1.0] * abs(prompt_token_len_2 -
            neg_prompt_token_len_2)
    embeds = []
    neg_embeds = []
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy(), prompt_weights.copy())
    neg_prompt_token_groups, neg_prompt_weight_groups = (
        group_tokens_and_weights(neg_prompt_tokens.copy(),
        neg_prompt_weights.copy()))
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy(), prompt_weights_2.copy())
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = (
        group_tokens_and_weights(neg_prompt_tokens_2.copy(),
        neg_prompt_weights_2.copy()))
    for i in range(len(prompt_token_groups)):
        token_tensor = torch.tensor([prompt_token_groups[i]], dtype=torch.
            long, device=device)
        weight_tensor = torch.tensor(prompt_weight_groups[i], dtype=torch.
            float16, device=device)
        token_tensor_2 = torch.tensor([prompt_token_groups_2[i]], dtype=
            torch.long, device=device)
        prompt_embeds_1 = pipe.text_encoder(token_tensor.to(device),
            output_hidden_states=True)
        prompt_embeds_2 = pipe.text_encoder_2(token_tensor_2.to(device),
            output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds_2[0]
        if clip_skip is None:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        else:
            prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-
                (clip_skip + 2)]
            prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-
                (clip_skip + 2)]
        prompt_embeds_list = [prompt_embeds_1_hidden_states,
            prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                token_embedding[j] = token_embedding[-1] + (token_embedding
                    [j] - token_embedding[-1]) * weight_tensor[j]
        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)
        neg_token_tensor = torch.tensor([neg_prompt_token_groups[i]], dtype
            =torch.long, device=device)
        neg_token_tensor_2 = torch.tensor([neg_prompt_token_groups_2[i]],
            dtype=torch.long, device=device)
        neg_weight_tensor = torch.tensor(neg_prompt_weight_groups[i], dtype
            =torch.float16, device=device)
        neg_prompt_embeds_1 = pipe.text_encoder(neg_token_tensor.to(device),
            output_hidden_states=True)
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[
            -2]
        neg_prompt_embeds_2 = pipe.text_encoder_2(neg_token_tensor_2.to(
            device), output_hidden_states=True)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[
            -2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]
        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states,
            neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1
            ).squeeze(0)
        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                neg_token_embedding[z] = neg_token_embedding[-1] + (
                    neg_token_embedding[z] - neg_token_embedding[-1]
                    ) * neg_weight_tensor[z]
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    prompt_embeds = torch.cat(embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt,
        seq_len, -1)
    seq_len = negative_prompt_embeds.shape[1]
    negative_prompt_embeds = negative_prompt_embeds.repeat(1,
        num_images_per_prompt, 1)
    negative_prompt_embeds = negative_prompt_embeds.view(bs_embed *
        num_images_per_prompt, seq_len, -1)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1,
        num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, -1)
    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1,
        num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, -1)
    return (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds)
