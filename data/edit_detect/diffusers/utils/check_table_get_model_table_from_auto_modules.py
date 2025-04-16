def get_model_table_from_auto_modules():
    """Generates an up-to-date model table from the content of the auto modules."""
    config_mapping_names = (diffusers_module.models.auto.configuration_auto
        .CONFIG_MAPPING_NAMES)
    model_name_to_config = {name: config_mapping_names[code] for code, name in
        diffusers_module.MODEL_NAMES_MAPPING.items() if code in
        config_mapping_names}
    model_name_to_prefix = {name: config.replace('ConfigMixin', '') for 
        name, config in model_name_to_config.items()}
    slow_tokenizers = collections.defaultdict(bool)
    fast_tokenizers = collections.defaultdict(bool)
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)
    for attr_name in dir(diffusers_module):
        lookup_dict = None
        if attr_name.endswith('Tokenizer'):
            lookup_dict = slow_tokenizers
            attr_name = attr_name[:-9]
        elif attr_name.endswith('TokenizerFast'):
            lookup_dict = fast_tokenizers
            attr_name = attr_name[:-13]
        elif _re_tf_models.match(attr_name) is not None:
            lookup_dict = tf_models
            attr_name = _re_tf_models.match(attr_name).groups()[0]
        elif _re_flax_models.match(attr_name) is not None:
            lookup_dict = flax_models
            attr_name = _re_flax_models.match(attr_name).groups()[0]
        elif _re_pt_models.match(attr_name) is not None:
            lookup_dict = pt_models
            attr_name = _re_pt_models.match(attr_name).groups()[0]
        if lookup_dict is not None:
            while len(attr_name) > 0:
                if attr_name in model_name_to_prefix.values():
                    lookup_dict[attr_name] = True
                    break
                attr_name = ''.join(camel_case_split(attr_name)[:-1])
    model_names = list(model_name_to_config.keys())
    model_names.sort(key=str.lower)
    columns = ['Model', 'Tokenizer slow', 'Tokenizer fast',
        'PyTorch support', 'TensorFlow support', 'Flax Support']
    widths = [(len(c) + 2) for c in columns]
    widths[0] = max([len(name) for name in model_names]) + 2
    table = '|' + '|'.join([_center_text(c, w) for c, w in zip(columns,
        widths)]) + '|\n'
    table += '|' + '|'.join([(':' + '-' * (w - 2) + ':') for w in widths]
        ) + '|\n'
    check = {(True): '✅', (False): '❌'}
    for name in model_names:
        prefix = model_name_to_prefix[name]
        line = [name, check[slow_tokenizers[prefix]], check[fast_tokenizers
            [prefix]], check[pt_models[prefix]], check[tf_models[prefix]],
            check[flax_models[prefix]]]
        table += '|' + '|'.join([_center_text(l, w) for l, w in zip(line,
            widths)]) + '|\n'
    return table
