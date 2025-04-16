@classmethod
def from_config(cls, cfg):
    return {'min_sizes': cfg.TEST.AUG.MIN_SIZES, 'max_size': cfg.TEST.AUG.
        MAX_SIZE, 'flip': cfg.TEST.AUG.FLIP}
