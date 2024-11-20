def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.shared = TFSharedEmbeddings(config.vocab_size, config.d_model,
        name='shared')
    with tf.compat.v1.variable_scope('shared') as shared_abs_scope_name:
        pass
    embed_tokens = TFWrappedEmbeddings(self.shared, abs_scope_name=
        shared_abs_scope_name)
    encoder_config = copy.deepcopy(config)
    encoder_config.use_cache = False
    self.encoder = TFT5MainLayer(encoder_config, embed_tokens, name='encoder')
