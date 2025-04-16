def set_truncation_and_padding(self, padding_strategy: PaddingStrategy,
    truncation_strategy: TruncationStrategy, max_length: int, stride: int,
    pad_to_multiple_of: Optional[int]):
    """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy (:class:`~transformers.tokenization_utils_base.PaddingStrategy`):
                The kind of padding that will be applied to the input
            truncation_strategy (:class:`~transformers.tokenization_utils_base.TruncationStrategy`):
                The kind of truncation that will be applied to the input
            max_length (:obj:`int`):
                The maximum size of a sequence.
            stride (:obj:`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        """
    if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
        self._tokenizer.enable_truncation(max_length, stride=stride,
            strategy=truncation_strategy.value)
    else:
        self._tokenizer.no_truncation()
    if padding_strategy != PaddingStrategy.DO_NOT_PAD:
        self._tokenizer.enable_padding(length=max_length if 
            padding_strategy == PaddingStrategy.MAX_LENGTH else None,
            direction=self.padding_side, pad_id=self.pad_token_id,
            pad_type_id=self.pad_token_type_id, pad_token=self.pad_token,
            pad_to_multiple_of=pad_to_multiple_of)
    else:
        self._tokenizer.no_padding()
