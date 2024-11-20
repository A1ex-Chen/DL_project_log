def tokenize(self, text: str, pair: Optional[str]=None, add_special_tokens:
    bool=False, **kwargs) ->List[str]:
    """
        Converts a string in a sequence of tokens, using the backend Rust tokenizer.

        Note that this method behave differently between fast and slow tokenizers:

            - in fast tokenizers (instances of :class:`~transformers.PreTrainedTokenizerFast`), this method will
              replace the unknown tokens with the :obj:`unk_token`,
            - in slow tokenizers (instances of :class:`~transformers.PreTrainedTokenizer`), this method keep unknown
              tokens unchanged.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            pair (:obj:`str`, `optional`):
                A second sequence to be encoded with the first.
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific encode method. See details in
                :meth:`~transformers.PreTrainedTokenizer.__call__`

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
    raise NotImplementedError
