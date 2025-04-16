def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1:
    Optional[List[int]]=None, already_has_special_tokens: bool=False) ->List[
    int]:
    """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
    if already_has_special_tokens:
        if token_ids_1 is not None:
            raise ValueError(
                'You should not supply a second sequence if the provided sequence of ids is already formatted with special tokens for the model.'
                )
        return list(map(lambda x: 1 if x in [self.sep_token_id, self.
            cls_token_id] else 0, token_ids_0))
    prefix_ones = [1] * len(self.prefix_tokens)
    suffix_ones = [1] * len(self.suffix_tokens)
    if token_ids_1 is None:
        return prefix_ones + [0] * len(token_ids_0) + suffix_ones
    return prefix_ones + [0] * len(token_ids_0) + [0] * len(token_ids_1
        ) + suffix_ones
