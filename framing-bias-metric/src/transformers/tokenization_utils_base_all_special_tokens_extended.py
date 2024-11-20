@property
def all_special_tokens_extended(self) ->List[Union[str, AddedToken]]:
    """
        :obj:`List[Union[str, tokenizers.AddedToken]]`: All the special tokens (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.)
        mapped to class attributes.

        Don't convert tokens of :obj:`tokenizers.AddedToken` type to string so they can be used to control more finely
        how special tokens are tokenized.
        """
    all_toks = []
    set_attr = self.special_tokens_map_extended
    for attr_value in set_attr.values():
        all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (
            list, tuple)) else [attr_value])
    all_toks = list(OrderedDict.fromkeys(all_toks))
    return all_toks
