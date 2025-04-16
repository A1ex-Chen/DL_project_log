def __init__(self, config, **model_args):
    super().__init__(config)
    self.local_config = model_args['local_config']
    self.copy_vocab = model_args['copy_vocab']
    self.attachable_index = model_args['attachable_index']
    self.fg_str_dict = model_args['fg_str_dict']
    self.model_dim = config.d_model
    self.shared = nn.Embedding(config.vocab_size, config.d_model)
    encoder_config = copy.deepcopy(config)
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = T5Stack(encoder_config, self.shared, visual_enable=self.
        local_config.enable_visual, visual_input_dim=self.local_config.
        roi_dim + self.local_config.box_dim, use_orginal_enc_pos_embs=self.
        local_config.use_orginal_enc_pos_embs)
    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = T5Stack(decoder_config, self.shared, use_mention_flag=
        self.local_config.use_mention_flag, mention_flag_num=self.
        local_config.mention_flag_state, use_mf_scalar=self.local_config.
        use_mf_scalar)
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    self.init_weights()
