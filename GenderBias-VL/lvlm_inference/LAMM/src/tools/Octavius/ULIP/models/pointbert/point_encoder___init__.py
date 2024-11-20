def __init__(self, config, **kwargs):
    super().__init__()
    self.config = config
    self.args = kwargs['args']
    self.trans_dim = config.trans_dim
    self.depth = config.depth
    self.drop_path_rate = config.drop_path_rate
    self.cls_dim = config.cls_dim
    self.num_heads = config.num_heads
    self.group_size = config.group_size
    self.num_group = config.num_group
    self.group_divider = Group(num_group=self.num_group, group_size=self.
        group_size)
    self.encoder_dims = config.encoder_dims
    self.encoder = Encoder(encoder_channel=self.encoder_dims)
    self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
    self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
    self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(
        128, self.trans_dim))
    dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]
    self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.
        depth, drop_path_rate=dpr, num_heads=self.num_heads)
    self.norm = nn.LayerNorm(self.trans_dim)
