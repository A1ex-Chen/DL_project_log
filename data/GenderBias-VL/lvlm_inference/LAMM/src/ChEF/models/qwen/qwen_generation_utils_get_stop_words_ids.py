def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == 'raw':
        stop_words_ids = [tokenizer.encode('Human:'), [tokenizer.eod_id]]
    elif chat_format == 'chatml':
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')
    return stop_words_ids
