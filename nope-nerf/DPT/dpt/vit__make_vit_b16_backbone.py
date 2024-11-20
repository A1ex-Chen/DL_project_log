def _make_vit_b16_backbone(model, features=[96, 192, 384, 768], size=[384, 
    384], hooks=[2, 5, 8, 11], vit_features=768, use_readout='ignore',
    start_index=1, enable_attention_hooks=False):
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1')
        )
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2')
        )
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3')
        )
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4')
        )
    pretrained.activations = activations
    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention('attn_1'))
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention('attn_2'))
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention('attn_3'))
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention('attn_4'))
        pretrained.attention = attention
    readout_oper = get_readout_oper(vit_features, features, use_readout,
        start_index)
    pretrained.act_postprocess1 = nn.Sequential(readout_oper[0], Transpose(
        1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[0],
        kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels
        =features[0], out_channels=features[0], kernel_size=4, stride=4,
        padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess2 = nn.Sequential(readout_oper[1], Transpose(
        1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[1],
        kernel_size=1, stride=1, padding=0), nn.ConvTranspose2d(in_channels
        =features[1], out_channels=features[1], kernel_size=2, stride=2,
        padding=0, bias=True, dilation=1, groups=1))
    pretrained.act_postprocess3 = nn.Sequential(readout_oper[2], Transpose(
        1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[2],
        kernel_size=1, stride=1, padding=0))
    pretrained.act_postprocess4 = nn.Sequential(readout_oper[3], Transpose(
        1, 2), nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[3],
        kernel_size=1, stride=1, padding=0), nn.Conv2d(in_channels=features
        [3], out_channels=features[3], kernel_size=3, stride=2, padding=1))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex,
        pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed,
        pretrained.model)
    return pretrained
