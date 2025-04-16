@configurable
def __init__(self, lang_encoder: nn.Module, in_channels,
    mask_classification=True, *, hidden_dim: int, dim_proj: int,
    num_queries: int, contxt_len: int, nheads: int, dim_feedforward: int,
    dec_layers: int, pre_norm: bool, mask_dim: int, task_switch: dict,
    enforce_input_project: bool, max_spatial_len: int, attn_arch: dict):
    """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
    super().__init__()
    assert mask_classification, 'Only support mask classification model'
    self.mask_classification = mask_classification
    N_steps = hidden_dim // 2
    self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
    self.num_heads = nheads
    self.num_layers = dec_layers
    self.contxt_len = contxt_len
    self.transformer_self_attention_layers = nn.ModuleList()
    self.transformer_cross_attention_layers = nn.ModuleList()
    self.transformer_ffn_layers = nn.ModuleList()
    for _ in range(self.num_layers):
        self.transformer_self_attention_layers.append(SelfAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before
            =pre_norm))
        self.transformer_cross_attention_layers.append(CrossAttentionLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before
            =pre_norm))
        self.transformer_ffn_layers.append(FFNLayer(d_model=hidden_dim,
            dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=
            pre_norm))
    self.decoder_norm = nn.LayerNorm(hidden_dim)
    self.num_queries = num_queries
    self.query_feat = nn.Embedding(num_queries, hidden_dim)
    self.query_embed = nn.Embedding(num_queries, hidden_dim)
    self.pn_indicator = nn.Embedding(2, hidden_dim)
    self.num_feature_levels = 3
    self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
    self.input_proj = nn.ModuleList()
    for _ in range(self.num_feature_levels):
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj.append(Conv2d(in_channels, hidden_dim,
                kernel_size=1))
            weight_init.c2_xavier_fill(self.input_proj[-1])
        else:
            self.input_proj.append(nn.Sequential())
    self.task_switch = task_switch
    self.query_index = {}
    self.lang_encoder = lang_encoder
    if self.task_switch['mask']:
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
    trunc_normal_(self.class_embed, std=0.02)
    if task_switch['bbox']:
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    if task_switch['spatial']:
        self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty
            (hidden_dim, hidden_dim)) for x in range(3)])
        trunc_normal_(self.mask_sptial_embed[0], std=0.02)
        trunc_normal_(self.mask_sptial_embed[1], std=0.02)
        trunc_normal_(self.mask_sptial_embed[2], std=0.02)
        self.max_spatial_len = max_spatial_len
        num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
        self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
        self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
    attn_arch['NUM_LAYERS'] = self.num_layers
    self.attention_data = AttentionDataStruct(attn_arch, task_switch)
