def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load('facebookresearch/WSL-Images',
        'resnext101_32x8d_wsl')
    return _make_resnet_backbone(resnet)
