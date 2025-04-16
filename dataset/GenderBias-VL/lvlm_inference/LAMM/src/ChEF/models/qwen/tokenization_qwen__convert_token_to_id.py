def _convert_token_to_id(self, token: Union[bytes, str]) ->int:
    """Converts a token to an id using the vocab, special tokens included"""
    if token in self.special_tokens:
        return self.special_tokens[token]
    if token in self.mergeable_ranks:
        return self.mergeable_ranks[token]
    raise ValueError('unknown token')
