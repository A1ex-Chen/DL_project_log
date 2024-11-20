def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=
    1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=
    False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate
    =0.0, norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True,
    use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling
    =True, init_scale=0.001, use_checkpoint=False):
    super().__init__()
    self.image_size = img_size
    self.num_classes = num_classes
    self.num_features = self.embed_dim = embed_dim
    self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
        in_chans=in_chans, embed_dim=embed_dim)
    num_patches = self.patch_embed.num_patches
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    if use_abs_pos_emb:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1,
            embed_dim))
    else:
        self.pos_embed = None
    self.pos_drop = nn.Dropout(p=drop_rate)
    if use_shared_rel_pos_bias:
        self.rel_pos_bias = RelativePositionBias(window_size=self.
            patch_embed.patch_shape, num_heads=num_heads)
    else:
        self.rel_pos_bias = None
    self.use_checkpoint = use_checkpoint
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    self.use_rel_pos_bias = use_rel_pos_bias
    self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=
        drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=
        norm_layer, init_values=init_values, window_size=self.patch_embed.
        patch_shape if use_rel_pos_bias else None) for i in range(depth)])
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=0.02)
    trunc_normal_(self.cls_token, std=0.02)
    self.apply(self._init_weights)
    self.fix_init_weight()
