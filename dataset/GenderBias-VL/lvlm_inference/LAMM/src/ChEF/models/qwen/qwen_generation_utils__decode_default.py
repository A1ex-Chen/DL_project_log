def _decode_default(tokens: List[int], *, stop_words: List[str], eod_words:
    List[str], tokenizer: PreTrainedTokenizer, raw_text_len: int, verbose:
    bool=False, return_end_reason: bool=False, errors: str='replace'):
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print('\nRaw Generate: ', trim_decode_tokens)
    end_reason = f'Gen length {len(tokens)}'
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, '').strip()
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f'Gen {eod_word!r}'
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print('\nEnd Reason:', end_reason)
        print('\nGenerate: ', trim_decode_tokens)
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens
