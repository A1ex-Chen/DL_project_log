def __init__(self, cfg):
    self.is_train = False
    self.tfm_gens = utils.build_transform_gen(cfg, self.is_train)
    self.img_format = cfg.INPUT.FORMAT
    assert not cfg.MODEL.LOAD_PROPOSALS, 'not supported yet'
