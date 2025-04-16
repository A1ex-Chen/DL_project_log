@property
def special_tokens_map_extended(self) ->Dict[str, Union[str, AddedToken,
    List[Union[str, AddedToken]]]]:
    """
        :obj:`Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary
        mapping special token class attributes (:obj:`cls_token`, :obj:`unk_token`, etc.) to their values
        (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.).

        Don't convert tokens of :obj:`tokenizers.AddedToken` type to string so they can be used to control more finely
        how special tokens are tokenized.
        """
    set_attr = {}
    for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
        attr_value = getattr(self, '_' + attr)
        if attr_value:
            set_attr[attr] = attr_value
    return set_attr
