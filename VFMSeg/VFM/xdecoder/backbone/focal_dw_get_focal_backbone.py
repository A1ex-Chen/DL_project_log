@register_backbone
def get_focal_backbone(cfg):
    focal = D2FocalNet(cfg['MODEL'], 224)
    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        logger.info(f'=> init from {filename}')
        with PathManager.open(filename, 'rb') as f:
            ckpt = torch.load(f)['model']
        focal.load_weights(ckpt, cfg['MODEL']['BACKBONE']['FOCAL'].get(
            'PRETRAINED_LAYERS', ['*']), cfg['VERBOSE'])
    return focal
