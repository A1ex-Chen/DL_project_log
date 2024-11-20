def _get_logits_processor(self, repetition_penalty: float,
    no_repeat_ngram_size: int, bad_words_ids: List[List[int]], min_length:
    int, eos_token_id: int, prefix_allowed_tokens_fn: Callable[[int, torch.
    Tensor], List[int]], num_beams: int) ->LogitsProcessorList:
    """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """
    repetition_penalty = (repetition_penalty if repetition_penalty is not
        None else self.config.repetition_penalty)
    no_repeat_ngram_size = (no_repeat_ngram_size if no_repeat_ngram_size is not
        None else self.config.no_repeat_ngram_size)
    bad_words_ids = (bad_words_ids if bad_words_ids is not None else self.
        config.bad_words_ids)
    min_length = (min_length if min_length is not None else self.config.
        min_length)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.eos_token_id)
    processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=
            repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids,
            eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn, num_beams))
    return processors
