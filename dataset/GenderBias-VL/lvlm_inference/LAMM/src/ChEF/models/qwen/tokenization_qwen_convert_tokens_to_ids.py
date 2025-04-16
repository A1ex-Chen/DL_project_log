def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes,
    str]]]) ->List[int]:
    ids = []
    if isinstance(tokens, (str, bytes)):
        if tokens in self.special_tokens:
            return self.special_tokens[tokens]
        else:
            return self.mergeable_ranks.get(tokens)
    for token in tokens:
        if token in self.special_tokens:
            ids.append(self.special_tokens[token])
        else:
            ids.append(self.mergeable_ranks.get(token))
    return ids
