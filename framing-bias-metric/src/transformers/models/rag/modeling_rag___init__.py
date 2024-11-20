def __init__(self, config: Optional[PretrainedConfig]=None,
    question_encoder: Optional[PreTrainedModel]=None, generator: Optional[
    PreTrainedModel]=None, retriever: Optional=None, **kwargs):
    assert config is not None or question_encoder is not None and generator is not None, 'Either a configuration or an encoder and a generator has to be provided.'
    if config is None:
        config = RagConfig.from_question_encoder_generator_configs(
            question_encoder.config, generator.config, **kwargs)
    super().__init__(config)
    self.rag = RagModel(config=config, question_encoder=question_encoder,
        generator=generator, retriever=retriever)
