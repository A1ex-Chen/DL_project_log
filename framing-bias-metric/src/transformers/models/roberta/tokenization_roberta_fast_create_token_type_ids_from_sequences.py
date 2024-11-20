def create_token_type_ids_from_sequences(self, token_ids_0: List[int],
    token_ids_1: Optional[List[int]]=None) ->List[int]:
    """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.
        """
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]
    if token_ids_1 is None:
        return len(cls + token_ids_0 + sep) * [0]
    return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
