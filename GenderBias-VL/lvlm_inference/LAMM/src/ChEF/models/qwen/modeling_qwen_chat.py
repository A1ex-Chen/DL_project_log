def chat(self, tokenizer: PreTrainedTokenizer, query: str, history:
    Optional[HistoryType], system: str='You are a helpful assistant.',
    append_history: bool=True, stream: Optional[bool]=_SENTINEL,
    stop_words_ids: Optional[List[List[int]]]=None, generation_config:
    Optional[GenerationConfig]=None, **kwargs) ->Tuple[str, HistoryType]:
    generation_config = (generation_config if generation_config is not None
         else self.generation_config)
    assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
    assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
    if history is None:
        history = []
    if stop_words_ids is None:
        stop_words_ids = []
    max_window_size = kwargs.get('max_window_size', None)
    if max_window_size is None:
        max_window_size = generation_config.max_window_size
    raw_text, context_tokens = make_context(tokenizer, query, history=
        history, system=system, max_window_size=max_window_size,
        chat_format=generation_config.chat_format)
    stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format,
        tokenizer))
    input_ids = torch.tensor([context_tokens]).to(self.device)
    outputs = self.generate(input_ids, stop_words_ids=stop_words_ids,
        return_dict_in_generate=False, generation_config=generation_config,
        **kwargs)
    response = decode_tokens(outputs[0], tokenizer, raw_text_len=len(
        raw_text), context_length=len(context_tokens), chat_format=
        generation_config.chat_format, verbose=False, errors='replace')
    if append_history:
        history.append((query, response))
    return response, history
