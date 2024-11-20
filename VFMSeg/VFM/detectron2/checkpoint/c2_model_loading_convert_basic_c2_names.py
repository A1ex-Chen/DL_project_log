def convert_basic_c2_names(original_keys):
    """
    Apply some basic name conversion to names in C2 weights.
    It only deals with typical backbone models.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [{'pred_b': 'linear_b', 'pred_w': 'linear_w'}.get(k, k) for
        k in layer_keys]
    layer_keys = [k.replace('_', '.') for k in layer_keys]
    layer_keys = [re.sub('\\.b$', '.bias', k) for k in layer_keys]
    layer_keys = [re.sub('\\.w$', '.weight', k) for k in layer_keys]
    layer_keys = [re.sub('bn\\.s$', 'norm.weight', k) for k in layer_keys]
    layer_keys = [re.sub('bn\\.bias$', 'norm.bias', k) for k in layer_keys]
    layer_keys = [re.sub('bn\\.rm', 'norm.running_mean', k) for k in layer_keys
        ]
    layer_keys = [re.sub('bn\\.running.mean$', 'norm.running_mean', k) for
        k in layer_keys]
    layer_keys = [re.sub('bn\\.riv$', 'norm.running_var', k) for k in
        layer_keys]
    layer_keys = [re.sub('bn\\.running.var$', 'norm.running_var', k) for k in
        layer_keys]
    layer_keys = [re.sub('bn\\.gamma$', 'norm.weight', k) for k in layer_keys]
    layer_keys = [re.sub('bn\\.beta$', 'norm.bias', k) for k in layer_keys]
    layer_keys = [re.sub('gn\\.s$', 'norm.weight', k) for k in layer_keys]
    layer_keys = [re.sub('gn\\.bias$', 'norm.bias', k) for k in layer_keys]
    layer_keys = [re.sub('^res\\.conv1\\.norm\\.', 'conv1.norm.', k) for k in
        layer_keys]
    layer_keys = [re.sub('^conv1\\.', 'stem.conv1.', k) for k in layer_keys]
    layer_keys = [k.replace('.branch1.', '.shortcut.') for k in layer_keys]
    layer_keys = [k.replace('.branch2a.', '.conv1.') for k in layer_keys]
    layer_keys = [k.replace('.branch2b.', '.conv2.') for k in layer_keys]
    layer_keys = [k.replace('.branch2c.', '.conv3.') for k in layer_keys]
    layer_keys = [re.sub('^body.conv.fcn', 'body_conv_fcn', k) for k in
        layer_keys]
    layer_keys = [k.replace('AnnIndex.lowres', 'ann_index_lowres') for k in
        layer_keys]
    layer_keys = [k.replace('Index.UV.lowres', 'index_uv_lowres') for k in
        layer_keys]
    layer_keys = [k.replace('U.lowres', 'u_lowres') for k in layer_keys]
    layer_keys = [k.replace('V.lowres', 'v_lowres') for k in layer_keys]
    return layer_keys
