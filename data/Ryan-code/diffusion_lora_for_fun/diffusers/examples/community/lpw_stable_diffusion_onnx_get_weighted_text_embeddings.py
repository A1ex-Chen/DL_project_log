def get_weighted_text_embeddings(pipe, prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]]=None,
    max_embeddings_multiples: Optional[int]=4, no_boseos_middle: Optional[
    bool]=False, skip_parsing: Optional[bool]=False, skip_weighting:
    Optional[bool]=False, **kwargs):
    """
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`OnnxStableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2
        ) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]
    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe,
            prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe,
                uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt,
            max_length=max_length, truncation=True, return_tensors='np').
            input_ids]
        prompt_weights = [([1.0] * len(token)) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [token[1:-1] for token in pipe.tokenizer(
                uncond_prompt, max_length=max_length, truncation=True,
                return_tensors='np').input_ids]
            uncond_weights = [([1.0] * len(token)) for token in uncond_tokens]
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in
            uncond_tokens]))
    max_embeddings_multiples = min(max_embeddings_multiples, (max_length - 
        1) // (pipe.tokenizer.model_max_length - 2) + 1)
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2
        ) * max_embeddings_multiples + 2
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = getattr(pipe.tokenizer, 'pad_token_id', eos)
    prompt_tokens, prompt_weights = pad_tokens_and_weights(prompt_tokens,
        prompt_weights, max_length, bos, eos, pad, no_boseos_middle=
        no_boseos_middle, chunk_length=pipe.tokenizer.model_max_length)
    prompt_tokens = np.array(prompt_tokens, dtype=np.int32)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(uncond_tokens,
            uncond_weights, max_length, bos, eos, pad, no_boseos_middle=
            no_boseos_middle, chunk_length=pipe.tokenizer.model_max_length)
        uncond_tokens = np.array(uncond_tokens, dtype=np.int32)
    text_embeddings = get_unweighted_text_embeddings(pipe, prompt_tokens,
        pipe.tokenizer.model_max_length, no_boseos_middle=no_boseos_middle)
    prompt_weights = np.array(prompt_weights, dtype=text_embeddings.dtype)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(pipe,
            uncond_tokens, pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle)
        uncond_weights = np.array(uncond_weights, dtype=uncond_embeddings.dtype
            )
    if not skip_parsing and not skip_weighting:
        previous_mean = text_embeddings.mean(axis=(-2, -1))
        text_embeddings *= prompt_weights[:, :, None]
        text_embeddings *= (previous_mean / text_embeddings.mean(axis=(-2, -1))
            )[:, None, None]
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.mean(axis=(-2, -1))
            uncond_embeddings *= uncond_weights[:, :, None]
            uncond_embeddings *= (previous_mean / uncond_embeddings.mean(
                axis=(-2, -1)))[:, None, None]
    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings
