def _make_pretrained_deitb16_384(pretrained, use_readout='ignore', hooks=
    None, enable_attention_hooks=False):
    model = timm.create_model('vit_deit_base_patch16_384', pretrained=
        pretrained)
    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(model, features=[96, 192, 384, 768],
        hooks=hooks, use_readout=use_readout, enable_attention_hooks=
        enable_attention_hooks)
