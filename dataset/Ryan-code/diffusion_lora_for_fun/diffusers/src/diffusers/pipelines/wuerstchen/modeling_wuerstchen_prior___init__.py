@register_to_config
def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16,
    dropout=0.1):
    super().__init__()
    self.c_r = c_r
    self.projection = nn.Conv2d(c_in, c, kernel_size=1)
    self.cond_mapper = nn.Sequential(nn.Linear(c_cond, c), nn.LeakyReLU(0.2
        ), nn.Linear(c, c))
    self.blocks = nn.ModuleList()
    for _ in range(depth):
        self.blocks.append(ResBlock(c, dropout=dropout))
        self.blocks.append(TimestepBlock(c, c_r))
        self.blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=
            dropout))
    self.out = nn.Sequential(WuerstchenLayerNorm(c, elementwise_affine=
        False, eps=1e-06), nn.Conv2d(c, c_in * 2, kernel_size=1))
    self.gradient_checkpointing = False
    self.set_default_attn_processor()
