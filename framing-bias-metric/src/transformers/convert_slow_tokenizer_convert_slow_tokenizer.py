def convert_slow_tokenizer(transformer_tokenizer) ->Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer (:class:`~transformers.tokenization_utils_base.PreTrainedTokenizer`):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`.

    Return:
        A instance of :class:`~tokenizers.Tokenizer` to be used as the backend tokenizer of a
        :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`
    """
    tokenizer_class_name = transformer_tokenizer.__class__.__name__
    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f'An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance. No converter was found. Currently available slow->fast convertors: {list(SLOW_TO_FAST_CONVERTERS.keys())}'
            )
    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]
    return converter_class(transformer_tokenizer).converted()
