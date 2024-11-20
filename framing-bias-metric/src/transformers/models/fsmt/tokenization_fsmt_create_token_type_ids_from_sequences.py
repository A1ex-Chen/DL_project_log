def create_token_type_ids_from_sequences(self, token_ids_0: List[int],
    token_ids_1: Optional[List[int]]=None) ->List[int]:
    """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).

        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An
        FAIRSEQ_TRANSFORMER sequence pair mask has the following format:
        """
    sep = [self.sep_token_id]
    if token_ids_1 is None:
        return len(token_ids_0 + sep) * [0]
    return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
