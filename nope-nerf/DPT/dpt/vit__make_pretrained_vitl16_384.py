def _make_pretrained_vitl16_384(pretrained, use_readout='ignore', hooks=
    None, enable_attention_hooks=False):
    model = timm.create_model('vit_large_patch16_384', pretrained=pretrained)
    hooks = [5, 11, 17, 23] if hooks == None else hooks
    return _make_vit_b16_backbone(model, features=[256, 512, 1024, 1024],
        hooks=hooks, vit_features=1024, use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks)
