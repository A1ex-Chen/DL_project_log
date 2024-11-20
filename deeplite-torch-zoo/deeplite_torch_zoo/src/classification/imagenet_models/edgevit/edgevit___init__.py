def __init__(self, depth=[1, 2, 5, 3], img_size=224, in_chans=3,
    num_classes=1000, embed_dim=[48, 96, 240, 384], head_dim=64, mlp_ratio=
    4.0, qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=
    0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None, sr_ratios
    =[4, 2, 2, 1], **kwargs):
    """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
        """
    super().__init__()
    self.num_classes = num_classes
    self.num_features = self.embed_dim = embed_dim
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
    self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4,
        in_chans=in_chans, embed_dim=embed_dim[0])
    self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2,
        in_chans=embed_dim[0], embed_dim=embed_dim[1])
    self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2,
        in_chans=embed_dim[1], embed_dim=embed_dim[2])
    self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2,
        in_chans=embed_dim[2], embed_dim=embed_dim[3])
    self.pos_drop = nn.Dropout(p=drop_rate)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
    num_heads = [(dim // head_dim) for dim in embed_dim]
    self.blocks1 = nn.ModuleList([LGLBlock(dim=embed_dim[0], num_heads=
        num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=
        qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i
        ], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(
        depth[0])])
    self.blocks2 = nn.ModuleList([LGLBlock(dim=embed_dim[1], num_heads=
        num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=
        qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i +
        depth[0]], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in
        range(depth[1])])
    self.blocks3 = nn.ModuleList([LGLBlock(dim=embed_dim[2], num_heads=
        num_heads[2], mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=
        qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i +
        depth[0] + depth[1]], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for
        i in range(depth[2])])
    self.blocks4 = nn.ModuleList([LGLBlock(dim=embed_dim[3], num_heads=
        num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=
        qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i +
        depth[0] + depth[1] + depth[2]], norm_layer=norm_layer, sr_ratio=
        sr_ratios[3]) for i in range(depth[3])])
    self.norm = nn.BatchNorm2d(embed_dim[-1])
    if representation_size:
        self.num_features = representation_size
        self.pre_logits = nn.Sequential(OrderedDict([('fc', nn.Linear(
            embed_dim, representation_size)), ('act', nn.Tanh())]))
    else:
        self.pre_logits = nn.Identity()
    self.head = nn.Linear(embed_dim[-1], num_classes
        ) if num_classes > 0 else nn.Identity()
    self.apply(self._init_weights)
