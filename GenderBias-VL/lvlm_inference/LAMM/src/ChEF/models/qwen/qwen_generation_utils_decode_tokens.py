def decode_tokens(tokens: Union[torch.LongTensor, TokensType], tokenizer:
    PreTrainedTokenizer, raw_text_len: int, context_length: int,
    chat_format: str, verbose: bool=False, return_end_reason: bool=False,
    errors: str='replace') ->str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()
    if chat_format == 'chatml':
        return _decode_chatml(tokens, stop_words=[], eod_token_ids=[
            tokenizer.im_start_id, tokenizer.im_end_id], tokenizer=
            tokenizer, raw_text_len=raw_text_len, context_length=
            context_length, verbose=verbose, return_end_reason=
            return_end_reason, errors=errors)
    elif chat_format == 'raw':
        return _decode_default(tokens, stop_words=['<|endoftext|>'],
            eod_words=['<|endoftext|>'], tokenizer=tokenizer, raw_text_len=
            raw_text_len, verbose=verbose, return_end_reason=
            return_end_reason, errors=errors)
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')
