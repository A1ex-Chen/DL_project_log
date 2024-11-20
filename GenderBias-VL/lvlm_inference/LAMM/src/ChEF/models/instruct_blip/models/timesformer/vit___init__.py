def __init__(self, image_size=224, patch_size=16, n_frms=8, attn_drop_rate=
    0.0, drop_path_rate=0.1, drop_rate=0, use_grad_ckpt=False, ckpt_layer=0,
    remove_classifier=True, **kwargs):
    super(TimeSformer, self).__init__()
    self.img_size = image_size
    self.patch_size = patch_size
    self.num_frames = n_frms
    self.attn_drop_rate = attn_drop_rate
    self.drop_path_rate = drop_path_rate
    self.drop_rate = drop_rate
    self.use_grad_ckpt = use_grad_ckpt
    self.ckpt_layer = ckpt_layer
    self.attention_type = 'divided_space_time'
    logging.info(
        f'Initializing TimeSformer with img_size={self.img_size}, patch_size={self.patch_size}, num_frames={self.num_frames}'
        )
    self.num_classes = 400
    self.model = VisionTransformer(img_size=self.img_size, num_classes=self
        .num_classes, patch_size=self.patch_size, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.
        LayerNorm, eps=1e-06), drop_rate=self.drop_rate, attn_drop_rate=
        self.attn_drop_rate, drop_path_rate=self.drop_path_rate, num_frames
        =self.num_frames, attention_type=self.attention_type,
        use_grad_checkpointing=self.use_grad_ckpt, ckpt_layer=self.
        ckpt_layer, **kwargs)
    if remove_classifier:
        self.model.remove_classifier()
    self.model.default_cfg = default_cfgs['vit_base_patch' + str(self.
        patch_size) + '_224']
    self.num_patches = self.img_size // self.patch_size * (self.img_size //
        self.patch_size)
