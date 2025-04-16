def _decode_chatml(tokens: List[int], *, stop_words: List[str],
    eod_token_ids: List[int], tokenizer: PreTrainedTokenizer, raw_text_len:
    int, context_length: int, verbose: bool=False, return_end_reason: bool=
    False, errors: str='replace'):
    end_reason = f'Gen length {len(tokens)}'
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f'Gen {tokenizer.decode([tokens[eod_token_idx]])!r}'
            break
    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors
        )[raw_text_len:]
    if verbose:
        print('\nRaw Generate w/o EOD:', tokenizer.decode(tokens, errors=
            errors)[raw_text_len:])
        print('\nRaw Generate:', trim_decode_tokens)
        print('\nEnd Reason:', end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, '').strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print('\nGenerate:', trim_decode_tokens)
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens
