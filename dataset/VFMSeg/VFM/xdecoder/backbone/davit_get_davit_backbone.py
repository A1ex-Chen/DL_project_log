@register_backbone
def get_davit_backbone(cfg):
    davit = D2DaViT(cfg['MODEL'], 224)
    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        logger.info(f'=> init from {filename}')
        davit.from_pretrained(filename, cfg['MODEL']['BACKBONE']['DAVIT'].
            get('PRETRAINED_LAYERS', ['*']), cfg['VERBOSE'])
    return davit
