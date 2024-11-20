def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1:
    Optional[List[int]]=None, already_has_special_tokens: bool=False) ->List[
    int]:
    """Retrieve sequence ids from a token list that has no special tokens
        added. This method is called when adding special tokens using the
        tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
    if already_has_special_tokens:
        return super().get_special_tokens_mask(token_ids_0=token_ids_0,
            token_ids_1=token_ids_1, already_has_special_tokens=True)
    if token_ids_1 is None:
        return [1] + [0] * len(token_ids_0) + [1]
    return [1] + [0] * len(token_ids_0) + [1, 1] + [0] * len(token_ids_1) + [1]
