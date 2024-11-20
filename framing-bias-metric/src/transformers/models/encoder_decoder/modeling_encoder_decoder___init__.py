def __init__(self, config: Optional[PretrainedConfig]=None, encoder:
    Optional[PreTrainedModel]=None, decoder: Optional[PreTrainedModel]=None):
    assert config is not None or encoder is not None and decoder is not None, 'Either a configuration or an Encoder and a decoder has to be provided'
    if config is None:
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.
            config, decoder.config)
    else:
        assert isinstance(config, self.config_class
            ), 'config: {} has to be of type {}'.format(config, self.
            config_class)
    super().__init__(config)
    if encoder is None:
        from ..auto.modeling_auto import AutoModel
        encoder = AutoModel.from_config(config.encoder)
    if decoder is None:
        from ..auto.modeling_auto import AutoModelForCausalLM
        decoder = AutoModelForCausalLM.from_config(config.decoder)
    self.encoder = encoder
    self.decoder = decoder
    assert self.encoder.get_output_embeddings(
        ) is None, 'The encoder {} should not have a LM Head. Please use a model without LM Head'
    self.tie_weights()
