def build_inputs_with_special_tokens(self, token_ids_0: List[int],
    token_ids_1: Optional[List[int]]=None) ->List[int]:
    """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:

        - single sequence: ``X <sep> <cls>``
        - pair of sequences: ``A <sep> B <sep> <cls>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
    sep = [self.sep_token_id]
    cls = [self.cls_token_id]
    if token_ids_1 is None:
        return token_ids_0 + sep + cls
    return token_ids_0 + sep + token_ids_1 + sep + cls
