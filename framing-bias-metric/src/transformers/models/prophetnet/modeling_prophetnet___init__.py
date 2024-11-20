def __init__(self, config):
    super().__init__(config)
    config = copy.deepcopy(config)
    config.is_decoder = True
    config.is_encoder_decoder = False
    self.decoder = ProphetNetDecoder(config)
    self.padding_idx = config.pad_token_id
    self.disable_ngram_loss = config.disable_ngram_loss
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.init_weights()
