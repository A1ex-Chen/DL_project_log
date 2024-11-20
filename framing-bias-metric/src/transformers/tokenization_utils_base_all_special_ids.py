@property
def all_special_ids(self) ->List[int]:
    """
        :obj:`List[int]`: List the ids of the special tokens(:obj:`'<unk>'`, :obj:`'<cls>'`, etc.) mapped to class
        attributes.
        """
    all_toks = self.all_special_tokens
    all_ids = self.convert_tokens_to_ids(all_toks)
    return all_ids
