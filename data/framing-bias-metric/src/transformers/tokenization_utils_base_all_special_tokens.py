@property
def all_special_tokens(self) ->List[str]:
    """
        :obj:`List[str]`: All the special tokens (:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class attributes.

        Convert tokens of :obj:`tokenizers.AddedToken` type to string.
        """
    all_toks = [str(s) for s in self.all_special_tokens_extended]
    return all_toks
