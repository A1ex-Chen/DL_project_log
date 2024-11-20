@classmethod
def from_config(cls, cfg, input_shape):
    ret = super().from_config(cfg, input_shape)
    ret.pop('proposal_matcher')
    return ret
