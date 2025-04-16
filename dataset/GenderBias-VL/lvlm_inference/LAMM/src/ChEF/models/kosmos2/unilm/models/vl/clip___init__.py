def __init__(self, embed_dim, vision_cfg, text_cfg, quick_gelu=False):
    super().__init__()
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = QuickGELU if quick_gelu else nn.GELU
    if vision_cfg.timm_model_name:
        raise NotImplementedError
        self.visual = TimmModel(vision_cfg.timm_model_name, pretrained=
            vision_cfg.timm_model_pretrained, pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj, embed_dim=embed_dim, image_size=
            vision_cfg.image_size)
        act_layer = nn.GELU
    elif isinstance(vision_cfg.layers, (tuple, list)):
        raise NotImplementedError
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        self.visual = ModifiedResNet(layers=vision_cfg.layers, output_dim=
            embed_dim, heads=vision_heads, image_size=vision_cfg.image_size,
            width=vision_cfg.width)
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer4Seq2Seq(image_size=vision_cfg.
            image_size, patch_size=vision_cfg.patch_size, width=vision_cfg.
            width, layers=vision_cfg.layers, heads=vision_heads, mlp_ratio=
            vision_cfg.mlp_ratio, output_dim=embed_dim, act_layer=act_layer)
    self.init_parameters()
