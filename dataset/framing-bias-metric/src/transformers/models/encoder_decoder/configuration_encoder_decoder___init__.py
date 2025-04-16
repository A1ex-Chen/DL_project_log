def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert 'encoder' in kwargs and 'decoder' in kwargs, 'Config has to be initialized with encoder and decoder config'
    encoder_config = kwargs.pop('encoder')
    encoder_model_type = encoder_config.pop('model_type')
    decoder_config = kwargs.pop('decoder')
    decoder_model_type = decoder_config.pop('model_type')
    from ..auto.configuration_auto import AutoConfig
    self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
    self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
    self.is_encoder_decoder = True
