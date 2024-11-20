def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
    add_prefix_space = kwargs.pop('add_prefix_space', self.add_prefix_space)
    if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not
        text[0].isspace()):
        text = ' ' + text
    return text, kwargs
