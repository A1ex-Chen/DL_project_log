def extension_from_parameters(params, framework):
    """Construct string for saving model with annotation of parameters"""
    ext = framework
    ext += '.A={}'.format(params['activation'])
    ext += '.B={}'.format(params['batch_size'])
    ext += '.D={}'.format(params['dropout'])
    ext += '.E={}'.format(params['epochs'])
    if params['feature_subsample']:
        ext += '.F={}'.format(params['feature_subsample'])
    if 'cell_noise_sigma' in params:
        ext += '.N={}'.format(params['cell_noise_sigma'])
    if 'conv' in params:
        name = 'LC' if 'locally_connected' in params else 'C'
        layer_list = list(range(0, len(params['conv'])))
        for layer, i in enumerate(layer_list):
            filters = params['conv'][i][0]
            filter_len = params['conv'][i][1]
            stride = params['conv'][i][2]
            if filters <= 0 or filter_len <= 0 or stride <= 0:
                break
            ext += '.{}{}={},{},{}'.format(name, layer + 1, filters,
                filter_len, stride)
        if 'pool' in params and params['conv'][0] and params['conv'][1]:
            ext += '.P={}'.format(params['pool'])
    if 'dense' in params:
        for i, n in enumerate(params['dense']):
            if n:
                ext += '.D{}={}'.format(i + 1, n)
    if params['batch_normalization']:
        ext += '.BN'
    ext += '.S={}'.format(params['scaling'])
    return ext
