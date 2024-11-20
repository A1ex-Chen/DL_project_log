@classmethod
def from_question_encoder_generator_configs(cls, question_encoder_config:
    PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) ->PretrainedConfig:
    """
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
    return cls(question_encoder=question_encoder_config.to_dict(),
        generator=generator_config.to_dict(), **kwargs)
