def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]],
    special_tokens: bool=False) ->int:
    if not special_tokens and new_tokens:
        raise ValueError('Adding regular tokens is not supported')
    for token in new_tokens:
        surface_form = token.content if isinstance(token, AddedToken
            ) else token
        if surface_form not in SPECIAL_TOKENS + self.IMAGE_ST:
            raise ValueError('Adding unknown special tokens is not supported')
    return 0
