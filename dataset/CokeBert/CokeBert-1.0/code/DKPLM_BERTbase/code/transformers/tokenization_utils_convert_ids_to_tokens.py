def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
    """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
    if isinstance(ids, int):
        if ids in self.added_tokens_decoder:
            return self.added_tokens_decoder[ids]
        else:
            return self._convert_id_to_token(ids)
    tokens = []
    for index in ids:
        if skip_special_tokens and index in self.all_special_ids:
            continue
        if index in self.added_tokens_decoder:
            tokens.append(self.added_tokens_decoder[index])
        else:
            tokens.append(self._convert_id_to_token(index))
    return tokens
