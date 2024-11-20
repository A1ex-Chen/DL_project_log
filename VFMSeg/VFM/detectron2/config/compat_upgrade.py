@classmethod
def upgrade(cls, cfg: CN) ->None:
    super().upgrade(cfg)
    if cfg.MODEL.META_ARCHITECTURE == 'RetinaNet':
        _rename(cfg, 'MODEL.RETINANET.ANCHOR_ASPECT_RATIOS',
            'MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS')
        _rename(cfg, 'MODEL.RETINANET.ANCHOR_SIZES',
            'MODEL.ANCHOR_GENERATOR.SIZES')
        del cfg['MODEL']['RPN']['ANCHOR_SIZES']
        del cfg['MODEL']['RPN']['ANCHOR_ASPECT_RATIOS']
    else:
        _rename(cfg, 'MODEL.RPN.ANCHOR_ASPECT_RATIOS',
            'MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS')
        _rename(cfg, 'MODEL.RPN.ANCHOR_SIZES', 'MODEL.ANCHOR_GENERATOR.SIZES')
        del cfg['MODEL']['RETINANET']['ANCHOR_SIZES']
        del cfg['MODEL']['RETINANET']['ANCHOR_ASPECT_RATIOS']
    del cfg['MODEL']['RETINANET']['ANCHOR_STRIDES']
