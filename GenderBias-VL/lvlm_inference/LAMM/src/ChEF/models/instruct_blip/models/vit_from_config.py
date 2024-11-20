@classmethod
def from_config(cls, cfg, from_pretrained=False):
    vit_type = cfg.get('vit_type', 'base')
    image_size = cfg.get('image_size', 384)
    ckpt_layer = cfg.get('vit_ckpt_layer', 0)
    drop_path_rate = cfg.get('vit_drop_path_rate', 0)
    norm_layer_eps = cfg.get('vit_layer_norm_epsilon', -1)
    use_grad_checkpointing = cfg.get('vit_grad_ckpt', False)
    if norm_layer_eps == -1:
        norm_layer = None
    else:
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)
    assert vit_type in ['base', 'large'], 'vit parameter must be base or large'
    if vit_type == 'base':
        vision_width = 768
        visual_encoder = cls(img_size=image_size, patch_size=16, embed_dim=
            vision_width, depth=12, num_heads=12, use_grad_checkpointing=
            use_grad_checkpointing, ckpt_layer=ckpt_layer, drop_path_rate=0 or
            drop_path_rate, norm_layer=norm_layer)
        if from_pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(url=
                'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
                , map_location='cpu', check_hash=True)
            state_dict = checkpoint['model']
            state_dict['pos_embed'] = interpolate_pos_embed(state_dict[
                'pos_embed'], visual_encoder)
            msg = visual_encoder.load_state_dict(state_dict, strict=False)
    elif vit_type == 'large':
        vision_width = 1024
        visual_encoder = cls(img_size=image_size, patch_size=16, embed_dim=
            vision_width, depth=24, num_heads=16, use_grad_checkpointing=
            use_grad_checkpointing, ckpt_layer=ckpt_layer, drop_path_rate=
            0.1 or drop_path_rate, norm_layer=norm_layer)
        if from_pretrained:
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(visual_encoder, default_cfgs[
                'vit_large_patch16_224_in21k'])
    visual_encoder.vision_width = vision_width
    return visual_encoder
