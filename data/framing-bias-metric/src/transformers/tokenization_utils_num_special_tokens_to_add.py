def num_special_tokens_to_add(self, pair: bool=False) ->int:
    """
        Returns the number of added tokens when encoding a sequence with special tokens.

        .. note::
            This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not
            put this inside your training loop.

        Args:
            pair (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            :obj:`int`: Number of special tokens added to sequences.
        """
    token_ids_0 = []
    token_ids_1 = []
    return len(self.build_inputs_with_special_tokens(token_ids_0, 
        token_ids_1 if pair else None))
