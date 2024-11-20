def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None
    ) ->List[int]:
    """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens. A PEGASUS sequence has the following format, where ``X`` represents the sequence:

        - single sequence: ``X </s>``
        - pair of sequences: ``A B </s>`` (not intended use)

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
    if token_ids_1 is None:
        return token_ids_0 + [self.eos_token_id]
    return token_ids_0 + token_ids_1 + [self.eos_token_id]
