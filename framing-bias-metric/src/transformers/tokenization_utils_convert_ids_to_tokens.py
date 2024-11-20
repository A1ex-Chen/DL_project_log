def convert_ids_to_tokens(self, ids: Union[int, List[int]],
    skip_special_tokens: bool=False) ->Union[str, List[str]]:
    """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (:obj:`int` or :obj:`List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            :obj:`str` or :obj:`List[str]`: The decoded token(s).
        """
    if isinstance(ids, int):
        if ids in self.added_tokens_decoder:
            return self.added_tokens_decoder[ids]
        else:
            return self._convert_id_to_token(ids)
    tokens = []
    for index in ids:
        index = int(index)
        if skip_special_tokens and index in self.all_special_ids:
            continue
        if index in self.added_tokens_decoder:
            tokens.append(self.added_tokens_decoder[index])
        else:
            tokens.append(self._convert_id_to_token(index))
    return tokens
