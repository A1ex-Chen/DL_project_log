def fpn_map(name):
    """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
    splits = name.split('.')
    norm = '.norm' if 'norm' in splits else ''
    if name.startswith('fpn.inner.'):
        stage = int(splits[2][len('res'):])
        return 'fpn_lateral{}{}.{}'.format(stage, norm, splits[-1])
    elif name.startswith('fpn.res'):
        stage = int(splits[1][len('res'):])
        return 'fpn_output{}{}.{}'.format(stage, norm, splits[-1])
    return name
