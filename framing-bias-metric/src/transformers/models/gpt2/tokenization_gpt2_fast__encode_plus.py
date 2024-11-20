def _encode_plus(self, *args, **kwargs) ->BatchEncoding:
    is_split_into_words = kwargs.get('is_split_into_words', False)
    assert self.add_prefix_space or not is_split_into_words, f'You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with pretokenized inputs.'
    return super()._encode_plus(*args, **kwargs)
