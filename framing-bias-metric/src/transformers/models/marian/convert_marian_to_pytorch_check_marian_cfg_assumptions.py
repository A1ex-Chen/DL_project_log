def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {'tied-embeddings-all': True, 'layer-normalization':
        False, 'right-left': False, 'transformer-ffn-depth': 2,
        'transformer-aan-depth': 2, 'transformer-no-projection': False,
        'transformer-postprocess-emb': 'd', 'transformer-postprocess':
        'dan', 'transformer-preprocess': '', 'type': 'transformer',
        'ulr-dim-emb': 0, 'dec-cell-base-depth': 2, 'dec-cell-high-depth': 
        1, 'transformer-aan-nogate': False}
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        assert actual == v, f'Unexpected config value for {k} expected {v} got {actual}'
    check_equal(marian_cfg, 'transformer-ffn-activation',
        'transformer-aan-activation')
    check_equal(marian_cfg, 'transformer-ffn-depth', 'transformer-aan-depth')
    check_equal(marian_cfg, 'transformer-dim-ffn', 'transformer-dim-aan')
