def get_phrase_indices(self, prompt, phrases, token_map=None,
    add_suffix_if_not_found=False, verbose=False):
    for obj in phrases:
        if obj not in prompt:
            prompt += '| ' + obj
    if token_map is None:
        token_map = self.get_token_map(prompt=prompt, padding='do_not_pad',
            verbose=verbose)
    token_map_str = ' '.join(token_map)
    phrase_indices = []
    for obj in phrases:
        phrase_token_map = self.get_token_map(prompt=obj, padding=
            'do_not_pad', verbose=verbose)
        phrase_token_map = phrase_token_map[1:-1]
        phrase_token_map_len = len(phrase_token_map)
        phrase_token_map_str = ' '.join(phrase_token_map)
        if verbose:
            logger.info('Full str:', token_map_str, 'Substr:',
                phrase_token_map_str, 'Phrase:', phrases)
        obj_first_index = len(token_map_str[:token_map_str.index(
            phrase_token_map_str) - 1].split(' '))
        obj_position = list(range(obj_first_index, obj_first_index +
            phrase_token_map_len))
        phrase_indices.append(obj_position)
    if add_suffix_if_not_found:
        return phrase_indices, prompt
    return phrase_indices
