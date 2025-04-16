def create_controller(prompts: List[str], cross_attention_kwargs: Dict,
    num_inference_steps: int, tokenizer, device) ->AttentionControl:
    edit_type = cross_attention_kwargs.get('edit_type', None)
    local_blend_words = cross_attention_kwargs.get('local_blend_words', None)
    equalizer_words = cross_attention_kwargs.get('equalizer_words', None)
    equalizer_strengths = cross_attention_kwargs.get('equalizer_strengths',
        None)
    n_cross_replace = cross_attention_kwargs.get('n_cross_replace', 0.4)
    n_self_replace = cross_attention_kwargs.get('n_self_replace', 0.4)
    if edit_type == 'replace' and local_blend_words is None:
        return AttentionReplace(prompts, num_inference_steps,
            n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device
            )
    if edit_type == 'replace' and local_blend_words is not None:
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer,
            device=device)
        return AttentionReplace(prompts, num_inference_steps,
            n_cross_replace, n_self_replace, lb, tokenizer=tokenizer,
            device=device)
    if edit_type == 'refine' and local_blend_words is None:
        return AttentionRefine(prompts, num_inference_steps,
            n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device
            )
    if edit_type == 'refine' and local_blend_words is not None:
        lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer,
            device=device)
        return AttentionRefine(prompts, num_inference_steps,
            n_cross_replace, n_self_replace, lb, tokenizer=tokenizer,
            device=device)
    if edit_type == 'reweight':
        assert equalizer_words is not None and equalizer_strengths is not None, 'To use reweight edit, please specify equalizer_words and equalizer_strengths.'
        assert len(equalizer_words) == len(equalizer_strengths
            ), 'equalizer_words and equalizer_strengths must be of same length.'
        equalizer = get_equalizer(prompts[1], equalizer_words,
            equalizer_strengths, tokenizer=tokenizer)
        return AttentionReweight(prompts, num_inference_steps,
            n_cross_replace, n_self_replace, tokenizer=tokenizer, device=
            device, equalizer=equalizer)
    raise ValueError(
        f'Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.'
        )
