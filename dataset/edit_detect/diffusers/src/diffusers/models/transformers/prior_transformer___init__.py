@register_to_config
def __init__(self, num_attention_heads: int=32, attention_head_dim: int=64,
    num_layers: int=20, embedding_dim: int=768, num_embeddings=77,
    additional_embeddings=4, dropout: float=0.0, time_embed_act_fn: str=
    'silu', norm_in_type: Optional[str]=None, embedding_proj_norm_type:
    Optional[str]=None, encoder_hid_proj_type: Optional[str]='linear',
    added_emb_type: Optional[str]='prd', time_embed_dim: Optional[int]=None,
    embedding_proj_dim: Optional[int]=None, clip_embed_dim: Optional[int]=None
    ):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    self.additional_embeddings = additional_embeddings
    time_embed_dim = time_embed_dim or inner_dim
    embedding_proj_dim = embedding_proj_dim or embedding_dim
    clip_embed_dim = clip_embed_dim or embedding_dim
    self.time_proj = Timesteps(inner_dim, True, 0)
    self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim,
        out_dim=inner_dim, act_fn=time_embed_act_fn)
    self.proj_in = nn.Linear(embedding_dim, inner_dim)
    if embedding_proj_norm_type is None:
        self.embedding_proj_norm = None
    elif embedding_proj_norm_type == 'layer':
        self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim)
    else:
        raise ValueError(
            f'unsupported embedding_proj_norm_type: {embedding_proj_norm_type}'
            )
    self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim)
    if encoder_hid_proj_type is None:
        self.encoder_hidden_states_proj = None
    elif encoder_hid_proj_type == 'linear':
        self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
    else:
        raise ValueError(
            f'unsupported encoder_hid_proj_type: {encoder_hid_proj_type}')
    self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings +
        additional_embeddings, inner_dim))
    if added_emb_type == 'prd':
        self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
    elif added_emb_type is None:
        self.prd_embedding = None
    else:
        raise ValueError(
            f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, num_attention_heads, attention_head_dim, dropout=dropout,
        activation_fn='gelu', attention_bias=True) for d in range(num_layers)])
    if norm_in_type == 'layer':
        self.norm_in = nn.LayerNorm(inner_dim)
    elif norm_in_type is None:
        self.norm_in = None
    else:
        raise ValueError(f'Unsupported norm_in_type: {norm_in_type}.')
    self.norm_out = nn.LayerNorm(inner_dim)
    self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim)
    causal_attention_mask = torch.full([num_embeddings +
        additional_embeddings, num_embeddings + additional_embeddings], -
        10000.0)
    causal_attention_mask.triu_(1)
    causal_attention_mask = causal_attention_mask[None, ...]
    self.register_buffer('causal_attention_mask', causal_attention_mask,
        persistent=False)
    self.clip_mean = nn.Parameter(torch.zeros(1, clip_embed_dim))
    self.clip_std = nn.Parameter(torch.zeros(1, clip_embed_dim))
