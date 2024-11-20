def chat_stream(self, tokenizer: PreTrainedTokenizer, query: str, history:
    Optional[HistoryType], system: str='You are a helpful assistant.',
    stop_words_ids: Optional[List[List[int]]]=None, logits_processor:
    Optional[LogitsProcessorList]=None, generation_config: Optional[
    GenerationConfig]=None, **kwargs) ->Generator[str, Any, None]:
    generation_config = (generation_config if generation_config is not None
         else self.generation_config)
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
    if stop_words_ids is not None:
        stop_words_logits_processor = StopWordsLogitsProcessor(stop_words_ids
            =stop_words_ids, eos_token_id=generation_config.eos_token_id)
        if logits_processor is None:
            logits_processor = LogitsProcessorList([
                stop_words_logits_processor])
        else:
            logits_processor.append(stop_words_logits_processor)
    input_ids = torch.tensor([context_tokens]).to(self.device)
    from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
    self.__class__.generate_stream = NewGenerationMixin.generate
    self.__class__.sample_stream = NewGenerationMixin.sample_stream
    stream_config = StreamGenerationConfig(**generation_config.to_dict(),
        do_stream=True)

    def stream_generator():
        outputs = []
        for token in self.generate_stream(input_ids,
            return_dict_in_generate=False, generation_config=stream_config,
            logits_processor=logits_processor, seed=-1, **kwargs):
            outputs.append(token.item())
            yield tokenizer.decode(outputs, skip_special_tokens=True,
                errors='ignore', keep_image_special=True)
    return stream_generator()
