def generate(self, inputs: Optional[torch.Tensor]=None, generation_config:
    Optional[GenerationConfig]=None, logits_processor: Optional[
    LogitsProcessorList]=None, stopping_criteria: Optional[
    StoppingCriteriaList]=None, prefix_allowed_tokens_fn: Optional[Callable
    [[int, torch.Tensor], List[int]]]=None, synced_gpus: Optional[bool]=
    None, assistant_model: Optional['PreTrainedModel']=None, streamer:
    Optional['BaseStreamer']=None, **kwargs) ->Union[GenerateOutput, torch.
    LongTensor]:
    generation_config = (generation_config if generation_config is not None
         else self.generation_config)
    stop_words_ids = kwargs.pop('stop_words_ids', None)
    if stop_words_ids is None and generation_config is not None:
        stop_words_ids = getattr(generation_config, 'stop_words_ids', None)
    if stop_words_ids is None:
        stop_words_ids = getattr(generation_config, 'stop_words_ids', None)
    if stop_words_ids is not None:
        stop_words_logits_processor = StopWordsLogitsProcessor(stop_words_ids
            =stop_words_ids, eos_token_id=generation_config.eos_token_id)
        if logits_processor is None:
            logits_processor = LogitsProcessorList([
                stop_words_logits_processor])
        else:
            logits_processor.append(stop_words_logits_processor)
    return super().generate(inputs, generation_config=generation_config,
        logits_processor=logits_processor, stopping_criteria=
        stopping_criteria, prefix_allowed_tokens_fn=
        prefix_allowed_tokens_fn, synced_gpus=synced_gpus, assistant_model=
        assistant_model, streamer=streamer, **kwargs)
