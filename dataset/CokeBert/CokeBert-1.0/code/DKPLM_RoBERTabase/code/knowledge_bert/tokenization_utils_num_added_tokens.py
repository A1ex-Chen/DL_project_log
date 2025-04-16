def num_added_tokens(self, pair=False):
    """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
    token_ids_0 = []
    token_ids_1 = []
    return len(self.build_inputs_with_special_tokens(token_ids_0, 
        token_ids_1 if pair else None))
