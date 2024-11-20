def __init__(self, point_encoder, **kwargs):
    super().__init__()
    kwargs = EasyDict(kwargs)
    self.context_length = kwargs.context_length
    self.vision_width = kwargs.vision_width
    self.visual = kwargs.vision_model
    self.transformer = Transformer(width=kwargs.transformer_width, layers=
        kwargs.transformer_layers, heads=kwargs.transformer_heads,
        attn_mask=self.build_attention_mask())
    self.vocab_size = kwargs.vocab_size
    self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.
        transformer_width)
    self.positional_embedding = nn.Parameter(torch.empty(self.
        context_length, kwargs.transformer_width))
    self.ln_final = LayerNorm(kwargs.transformer_width)
    self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width,
        kwargs.embed_dim))
    self.text_projection = nn.Parameter(torch.empty(kwargs.
        transformer_width, kwargs.embed_dim))
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.initialize_parameters()
    self.point_encoder = point_encoder
    self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 768))
    nn.init.normal_(self.pc_projection, std=768 ** -0.5)
