def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps:
    Union[float, Dict[str, Tuple[float, float]]], tokenizer, max_num_words=77):
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {'default_': cross_replace_steps}
    if 'default_' not in cross_replace_steps:
        cross_replace_steps['default_'] = 0.0, 1.0
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1,
        max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words,
            cross_replace_steps['default_'], i)
    for key, item in cross_replace_steps.items():
        if key != 'default_':
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in
                range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words,
                        item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) -
        1, 1, 1, max_num_words)
    return alpha_time_words
