def cfg2task(cfg):
    """Guess from YAML dictionary."""
    m = cfg['head'][-1][-2].lower()
    if m in {'classify', 'classifier', 'cls', 'fc'}:
        return 'classify'
    if 'detect' in m:
        return 'detect'
    if m == 'segment':
        return 'segment'
    if m == 'pose':
        return 'pose'
    if m == 'obb':
        return 'obb'
