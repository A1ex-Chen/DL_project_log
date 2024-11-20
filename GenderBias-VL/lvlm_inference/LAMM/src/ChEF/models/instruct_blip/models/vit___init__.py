def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=
    1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=
    True, qk_scale=None, representation_size=None, drop_rate=0.0,
    attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=None,
    use_grad_checkpointing=False, ckpt_layer=0):
    """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
    super().__init__()
    self.num_features = self.embed_dim = embed_dim
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
    self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
        in_chans=in_chans, embed_dim=embed_dim)
    num_patches = self.patch_embed.num_patches
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=
        drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=
        norm_layer, use_grad_checkpointing=use_grad_checkpointing and i >= 
        depth - ckpt_layer) for i in range(depth)])
    self.norm = norm_layer(embed_dim)
    trunc_normal_(self.pos_embed, std=0.02)
    trunc_normal_(self.cls_token, std=0.02)
    self.apply(self._init_weights)
