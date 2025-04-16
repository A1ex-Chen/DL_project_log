@property
def special_tokens_map(self):
    """ A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
    set_attr = {}
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
        attr_value = getattr(self, '_' + attr)
        if attr_value:
            set_attr[attr] = attr_value
    return set_attr
