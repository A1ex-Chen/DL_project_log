@classmethod
def from_config(cls, cfg):
    backbone = build_backbone(cfg)
    return {'backbone': backbone, 'proposal_generator':
        build_proposal_generator(cfg, backbone.output_shape()),
        'pixel_mean': cfg.MODEL.PIXEL_MEAN, 'pixel_std': cfg.MODEL.PIXEL_STD}
