def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[
    List]=None, already_has_special_tokens: bool=False) ->List[int]:
    """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
    if already_has_special_tokens:
        return self._special_token_mask(token_ids_0)
    elif token_ids_1 is None:
        return self._special_token_mask(token_ids_0) + [1]
    else:
        return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
