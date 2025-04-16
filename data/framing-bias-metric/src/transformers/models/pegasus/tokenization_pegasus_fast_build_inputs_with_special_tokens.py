def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None
    ) ->List[int]:
    """
        Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

        - single sequence: ``X </s>``
        - pair of sequences: ``A B </s>`` (not intended use)

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
    if token_ids_1 is None:
        return token_ids_0 + [self.eos_token_id]
    return token_ids_0 + token_ids_1 + [self.eos_token_id]
