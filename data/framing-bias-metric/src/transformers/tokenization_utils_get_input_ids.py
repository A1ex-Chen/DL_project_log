def get_input_ids(text):
    if isinstance(text, str):
        tokens = self.tokenize(text, **kwargs)
        return self.convert_tokens_to_ids(tokens)
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text
        [0], str):
        if is_split_into_words:
            tokens = list(itertools.chain(*(self.tokenize(t,
                is_split_into_words=True, **kwargs) for t in text)))
            return self.convert_tokens_to_ids(tokens)
        else:
            return self.convert_tokens_to_ids(text)
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text
        [0], int):
        return text
    else:
        raise ValueError(
            'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.'
            )
