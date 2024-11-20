@classmethod
def from_encoder_decoder_configs(cls, encoder_config: PretrainedConfig,
    decoder_config: PretrainedConfig, **kwargs) ->PretrainedConfig:
    """
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
    logger.info(
        'Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config'
        )
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.
        to_dict(), **kwargs)
