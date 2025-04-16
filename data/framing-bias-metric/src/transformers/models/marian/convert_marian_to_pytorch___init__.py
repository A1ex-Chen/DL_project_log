def __init__(self, source_dir):
    npz_path = find_model_file(source_dir)
    self.state_dict = np.load(npz_path)
    cfg = load_config_from_state_dict(self.state_dict)
    assert cfg['dim-vocabs'][0] == cfg['dim-vocabs'][1]
    assert 'Wpos' not in self.state_dict, 'Wpos key in state dictionary'
    self.state_dict = dict(self.state_dict)
    self.wemb, self.final_bias = add_emb_entries(self.state_dict['Wemb'],
        self.state_dict[BIAS_KEY], 1)
    self.pad_token_id = self.wemb.shape[0] - 1
    cfg['vocab_size'] = self.pad_token_id + 1
    self.state_keys = list(self.state_dict.keys())
    assert 'Wtype' not in self.state_dict, 'Wtype key in state dictionary'
    self._check_layer_entries()
    self.source_dir = source_dir
    self.cfg = cfg
    hidden_size, intermediate_shape = self.state_dict['encoder_l1_ffn_W1'
        ].shape
    assert hidden_size == cfg['dim-emb'
        ] == 512, f"Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched or not 512"
    decoder_yml = cast_marian_config(load_yaml(source_dir / 'decoder.yml'))
    check_marian_cfg_assumptions(cfg)
    self.hf_config = MarianConfig(vocab_size=cfg['vocab_size'],
        decoder_layers=cfg['dec-depth'], encoder_layers=cfg['enc-depth'],
        decoder_attention_heads=cfg['transformer-heads'],
        encoder_attention_heads=cfg['transformer-heads'], decoder_ffn_dim=
        cfg['transformer-dim-ffn'], encoder_ffn_dim=cfg[
        'transformer-dim-ffn'], d_model=cfg['dim-emb'], activation_function
        =cfg['transformer-aan-activation'], pad_token_id=self.pad_token_id,
        eos_token_id=0, bos_token_id=0, max_position_embeddings=cfg[
        'dim-emb'], scale_embedding=True, normalize_embedding='n' in cfg[
        'transformer-preprocess'], static_position_embeddings=not cfg[
        'transformer-train-position-embeddings'], dropout=0.1, num_beams=
        decoder_yml['beam-size'], decoder_start_token_id=self.pad_token_id,
        bad_words_ids=[[self.pad_token_id]], max_length=512)
