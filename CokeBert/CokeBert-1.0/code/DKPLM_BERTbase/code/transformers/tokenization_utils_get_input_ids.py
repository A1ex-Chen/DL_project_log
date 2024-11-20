def get_input_ids(text):
    if isinstance(text, six.string_types):
        return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text
        [0], six.string_types):
        return self.convert_tokens_to_ids(text)
    elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text
        [0], int):
        return text
    else:
        raise ValueError(
            'Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.'
            )
