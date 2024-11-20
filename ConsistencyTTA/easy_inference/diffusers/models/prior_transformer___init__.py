@register_to_config
def __init__(self, num_attention_heads: int=32, attention_head_dim: int=64,
    num_layers: int=20, embedding_dim: int=768, num_embeddings=77,
    additional_embeddings=4, dropout: float=0.0):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    self.additional_embeddings = additional_embeddings
    self.time_proj = Timesteps(inner_dim, True, 0)
    self.time_embedding = TimestepEmbedding(inner_dim, inner_dim)
    self.proj_in = nn.Linear(embedding_dim, inner_dim)
    self.embedding_proj = nn.Linear(embedding_dim, inner_dim)
    self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
    self.positional_embedding = nn.Parameter(torch.zeros(1, num_embeddings +
        additional_embeddings, inner_dim))
    self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, num_attention_heads, attention_head_dim, dropout=dropout,
        activation_fn='gelu', attention_bias=True) for d in range(num_layers)])
    self.norm_out = nn.LayerNorm(inner_dim)
    self.proj_to_clip_embeddings = nn.Linear(inner_dim, embedding_dim)
    causal_attention_mask = torch.full([num_embeddings +
        additional_embeddings, num_embeddings + additional_embeddings], -
        10000.0)
    causal_attention_mask.triu_(1)
    causal_attention_mask = causal_attention_mask[None, ...]
    self.register_buffer('causal_attention_mask', causal_attention_mask,
        persistent=False)
    self.clip_mean = nn.Parameter(torch.zeros(1, embedding_dim))
    self.clip_std = nn.Parameter(torch.zeros(1, embedding_dim))
