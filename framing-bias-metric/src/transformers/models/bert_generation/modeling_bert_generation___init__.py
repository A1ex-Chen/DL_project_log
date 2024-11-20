def __init__(self, config):
    super().__init__(config)
    if not config.is_decoder:
        logger.warn(
            'If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`'
            )
    self.bert = BertGenerationEncoder(config)
    self.lm_head = BertGenerationOnlyLMHead(config)
    self.init_weights()
