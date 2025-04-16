def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1:
    Optional[List[int]]=None, already_has_special_tokens: bool=False) ->List[
    int]:
    """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of ids of the second sequence.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
    assert already_has_special_tokens and token_ids_1 is None, 'You cannot use ``already_has_special_tokens=False`` with this tokenizer. Please use a slow (full python) tokenizer to activate this argument.Or set `return_special_token_mask=True` when calling the encoding method to get the special tokens mask in any tokenizer. '
    all_special_ids = self.all_special_ids
    special_tokens_mask = [(1 if token in all_special_ids else 0) for token in
        token_ids_0]
    return special_tokens_mask
