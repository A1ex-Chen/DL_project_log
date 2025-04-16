def build_augmenter_params(augmenter_name, cutout_const, translate_const,
    num_layers, magnitude, autoaugmentation_name):
    if augmenter_name is None or augmenter_name not in ['randaugment',
        'autoaugment']:
        return {}
    augmenter_params = {}
    if cutout_const is not None:
        augmenter_params['cutout_const'] = cutout_const
    if translate_const is not None:
        augmenter_params['translate_const'] = translate_const
    if augmenter_name == 'randaugment':
        if num_layers is not None:
            augmenter_params['num_layers'] = num_layers
        if magnitude is not None:
            augmenter_params['magnitude'] = magnitude
    if augmenter_name == 'autoaugment':
        if autoaugmentation_name is not None:
            augmenter_params['autoaugmentation_name'] = autoaugmentation_name
    return augmenter_params
