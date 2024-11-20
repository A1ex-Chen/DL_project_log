@property
def special_tokens_map(self) ->Dict[str, Union[str, List[str]]]:
    """
        :obj:`Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (:obj:`cls_token`,
        :obj:`unk_token`, etc.) to their values (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).

        Convert potential tokens of :obj:`tokenizers.AddedToken` type to string.
        """
    set_attr = {}
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
        attr_value = getattr(self, '_' + attr)
        if attr_value:
            set_attr[attr] = str(attr_value)
    return set_attr
