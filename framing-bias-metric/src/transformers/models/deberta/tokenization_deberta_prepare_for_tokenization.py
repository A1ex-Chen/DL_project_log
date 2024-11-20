def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
    add_prefix_space = kwargs.pop('add_prefix_space', False)
    if is_split_into_words or add_prefix_space:
        text = ' ' + text
    return text, kwargs
