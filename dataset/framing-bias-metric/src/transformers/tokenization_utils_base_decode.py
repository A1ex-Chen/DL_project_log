def decode(self, token_ids: Union[int, List[int], 'np.ndarray',
    'torch.Tensor', 'tf.Tensor'], skip_special_tokens: bool=False,
    clean_up_tokenization_spaces: bool=True, **kwargs) ->str:
    """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids (:obj:`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`str`: The decoded sentence.
        """
    token_ids = to_py_obj(token_ids)
    return self._decode(token_ids=token_ids, skip_special_tokens=
        skip_special_tokens, clean_up_tokenization_spaces=
        clean_up_tokenization_spaces, **kwargs)
