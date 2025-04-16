def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    assert v1 == v2, f'hparams {k1},{k2} differ: {v1} != {v2}'
