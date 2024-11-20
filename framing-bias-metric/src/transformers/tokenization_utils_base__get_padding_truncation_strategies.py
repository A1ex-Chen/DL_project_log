def _get_padding_truncation_strategies(self, padding=False, truncation=
    False, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs):
    """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
    old_truncation_strategy = kwargs.pop('truncation_strategy',
        'do_not_truncate')
    old_pad_to_max_length = kwargs.pop('pad_to_max_length', False)
    if max_length is not None and padding is False and truncation is False:
        if verbose:
            if not self.deprecation_warnings.get(
                'Truncation-not-explicitly-activated', False):
                logger.warning(
                    "Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`."
                    )
            self.deprecation_warnings['Truncation-not-explicitly-activated'
                ] = True
        truncation = 'longest_first'
    if padding is False and old_pad_to_max_length:
        if verbose:
            warnings.warn(
                "The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert)."
                , FutureWarning)
        if max_length is None:
            padding_strategy = PaddingStrategy.LONGEST
        else:
            padding_strategy = PaddingStrategy.MAX_LENGTH
    elif padding is not False:
        if padding is True:
            padding_strategy = PaddingStrategy.LONGEST
        elif not isinstance(padding, PaddingStrategy):
            padding_strategy = PaddingStrategy(padding)
        elif isinstance(padding, PaddingStrategy):
            padding_strategy = padding
    else:
        padding_strategy = PaddingStrategy.DO_NOT_PAD
    if truncation is False and old_truncation_strategy != 'do_not_truncate':
        if verbose:
            warnings.warn(
                "The `truncation_strategy` argument is deprecated and will be removed in a future version, use `truncation=True` to truncate examples to a max length. You can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific truncation strategy selected among `truncation='only_first'` (will only truncate the first sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs)."
                , FutureWarning)
        truncation_strategy = TruncationStrategy(old_truncation_strategy)
    elif truncation is not False:
        if truncation is True:
            truncation_strategy = TruncationStrategy.LONGEST_FIRST
        elif not isinstance(truncation, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation)
        elif isinstance(truncation, TruncationStrategy):
            truncation_strategy = truncation
    else:
        truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
    if max_length is None:
        if padding_strategy == PaddingStrategy.MAX_LENGTH:
            if self.model_max_length > LARGE_INTEGER:
                if verbose:
                    if not self.deprecation_warnings.get(
                        'Asking-to-pad-to-max_length', False):
                        logger.warning(
                            'Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no padding.'
                            )
                    self.deprecation_warnings['Asking-to-pad-to-max_length'
                        ] = True
                padding_strategy = PaddingStrategy.DO_NOT_PAD
            else:
                max_length = self.model_max_length
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
            if self.model_max_length > LARGE_INTEGER:
                if verbose:
                    if not self.deprecation_warnings.get(
                        'Asking-to-truncate-to-max_length', False):
                        logger.warning(
                            'Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.'
                            )
                    self.deprecation_warnings[
                        'Asking-to-truncate-to-max_length'] = True
                truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
            else:
                max_length = self.model_max_length
    if padding_strategy != PaddingStrategy.DO_NOT_PAD and (not self.
        pad_token or self.pad_token_id < 0):
        raise ValueError(
            "Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )
    if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and 
        padding_strategy != PaddingStrategy.DO_NOT_PAD and 
        pad_to_multiple_of is not None and max_length is not None and 
        max_length % pad_to_multiple_of != 0):
        raise ValueError(
            f'Truncation and padding are both activated but truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of}).'
            )
    return padding_strategy, truncation_strategy, max_length, kwargs
