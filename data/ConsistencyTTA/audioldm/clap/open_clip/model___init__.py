def __init__(self, embed_dim: int, audio_cfg: CLAPAudioCfp, text_cfg:
    CLAPTextCfg, quick_gelu: bool=False, enable_fusion: bool=False,
    fusion_type: str='None', joint_embed_shape: int=512, mlp_act: str='relu'):
    super().__init__()
    if isinstance(audio_cfg, dict):
        audio_cfg = CLAPAudioCfp(**audio_cfg)
    if isinstance(text_cfg, dict):
        text_cfg = CLAPTextCfg(**text_cfg)
    self.audio_cfg = audio_cfg
    self.text_cfg = text_cfg
    self.enable_fusion = enable_fusion
    self.fusion_type = fusion_type
    self.joint_embed_shape = joint_embed_shape
    self.mlp_act = mlp_act
    self.context_length = text_cfg.context_length
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if mlp_act == 'relu':
        mlp_act_layer = nn.ReLU()
    elif mlp_act == 'gelu':
        mlp_act_layer = nn.GELU()
    else:
        raise NotImplementedError
    if audio_cfg.model_type == 'PANN':
        self.audio_branch = create_pann_model(audio_cfg, enable_fusion,
            fusion_type)
    elif audio_cfg.model_type == 'HTSAT':
        self.audio_branch = create_htsat_model(audio_cfg, enable_fusion,
            fusion_type)
    else:
        logging.error(f'Model config for {audio_cfg.model_type} not found')
        raise RuntimeError(
            f'Model config for {audio_cfg.model_type} not found.')
    if text_cfg.model_type == 'transformer':
        self.text_branch = Transformer(width=text_cfg.width, layers=
            text_cfg.layers, heads=text_cfg.heads, act_layer=act_layer)
        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width
            )
        self.positional_embedding = nn.Parameter(torch.empty(self.
            context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)
        self.text_transform = MLPLayers(units=[self.joint_embed_shape, self
            .joint_embed_shape, self.joint_embed_shape], dropout=0.1)
        self.text_projection = nn.Sequential(nn.Linear(text_cfg.width, self
            .joint_embed_shape), mlp_act_layer, nn.Linear(self.
            joint_embed_shape, self.joint_embed_shape))
    elif text_cfg.model_type == 'bert':
        self.text_branch = BertModel.from_pretrained('bert-base-uncased')
        self.text_transform = MLPLayers(units=[self.joint_embed_shape, self
            .joint_embed_shape, self.joint_embed_shape], dropout=0.1)
        self.text_projection = nn.Sequential(nn.Linear(768, self.
            joint_embed_shape), mlp_act_layer, nn.Linear(self.
            joint_embed_shape, self.joint_embed_shape))
    elif text_cfg.model_type == 'roberta':
        self.text_branch = RobertaModel.from_pretrained('roberta-base')
        self.text_transform = MLPLayers(units=[self.joint_embed_shape, self
            .joint_embed_shape, self.joint_embed_shape], dropout=0.1)
        self.text_projection = nn.Sequential(nn.Linear(768, self.
            joint_embed_shape), mlp_act_layer, nn.Linear(self.
            joint_embed_shape, self.joint_embed_shape))
    elif text_cfg.model_type == 'bart':
        self.text_branch = BartModel.from_pretrained('facebook/bart-base')
        self.text_transform = MLPLayers(units=[self.joint_embed_shape, self
            .joint_embed_shape, self.joint_embed_shape], dropout=0.1)
        self.text_projection = nn.Sequential(nn.Linear(768, self.
            joint_embed_shape), mlp_act_layer, nn.Linear(self.
            joint_embed_shape, self.joint_embed_shape))
    else:
        logging.error(f'Model config for {text_cfg.model_type} not found')
        raise RuntimeError(f'Model config for {text_cfg.model_type} not found.'
            )
    self.text_branch_type = text_cfg.model_type
    self.audio_transform = MLPLayers(units=[self.joint_embed_shape, self.
        joint_embed_shape, self.joint_embed_shape], dropout=0.1)
    self.audio_projection = nn.Sequential(nn.Linear(embed_dim, self.
        joint_embed_shape), mlp_act_layer, nn.Linear(self.joint_embed_shape,
        self.joint_embed_shape))
    self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.register_buffer('attn_mask', self.build_attention_mask(),
        persistent=False)
    self.init_text_branch_parameters()
