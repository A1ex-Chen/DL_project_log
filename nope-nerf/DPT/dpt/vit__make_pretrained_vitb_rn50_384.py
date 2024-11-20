def _make_pretrained_vitb_rn50_384(pretrained, use_readout='ignore', hooks=
    None, use_vit_only=False, enable_attention_hooks=False):
    model = timm.create_model('vit_base_resnet50_384', pretrained=pretrained)
    hooks = [0, 1, 8, 11] if hooks == None else hooks
    return _make_vit_b_rn50_backbone(model, features=[256, 512, 768, 768],
        size=[384, 384], hooks=hooks, use_vit_only=use_vit_only,
        use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
