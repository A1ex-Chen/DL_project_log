def _add_tokens(self, new_tokens: List[Union[str, AddedToken]],
    special_tokens=False) ->int:
    if special_tokens:
        return self._tokenizer.add_special_tokens(new_tokens)
    return self._tokenizer.add_tokens(new_tokens)
