@classmethod
def _from_pretrained(cls, resolved_vocab_files,
    pretrained_model_name_or_path, init_configuration, *init_inputs, **kwargs):
    if ('tokenizer_file' not in resolved_vocab_files or 
        resolved_vocab_files['tokenizer_file'] is None
        ) and cls.slow_tokenizer_class is not None:
        slow_tokenizer = cls.slow_tokenizer_class._from_pretrained(copy.
            deepcopy(resolved_vocab_files), pretrained_model_name_or_path,
            copy.deepcopy(init_configuration), *init_inputs, **copy.
            deepcopy(kwargs))
    else:
        slow_tokenizer = None
    tokenizer_config_file = resolved_vocab_files.pop('tokenizer_config_file',
        None)
    if tokenizer_config_file is not None:
        with open(tokenizer_config_file, encoding='utf-8'
            ) as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
        saved_init_inputs = init_kwargs.pop('init_inputs', ())
        if not init_inputs:
            init_inputs = saved_init_inputs
    else:
        init_kwargs = init_configuration
    init_kwargs.update(kwargs)

    def convert_added_tokens(obj: Union[AddedToken, Any]):
        if isinstance(obj, dict) and '__type' in obj and obj['__type'
            ] == 'AddedToken':
            obj.pop('__type')
            return AddedToken(**obj)
        elif isinstance(obj, (list, tuple)):
            return list(convert_added_tokens(o) for o in obj)
        elif isinstance(obj, dict):
            return {k: convert_added_tokens(v) for k, v in obj.items()}
        return obj
    init_kwargs = convert_added_tokens(init_kwargs)
    if pretrained_model_name_or_path in cls.max_model_input_sizes:
        model_max_length = cls.max_model_input_sizes[
            pretrained_model_name_or_path]
        if model_max_length is not None and isinstance(model_max_length, (
            int, float)):
            init_kwargs['model_max_length'] = min(init_kwargs.get(
                'model_max_length', int(1e+30)), model_max_length)
    added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
    for args_name, file_path in resolved_vocab_files.items():
        if args_name not in init_kwargs:
            init_kwargs[args_name] = file_path
    if slow_tokenizer is not None:
        init_kwargs['__slow_tokenizer'] = slow_tokenizer
    init_kwargs['name_or_path'] = pretrained_model_name_or_path
    try:
        tokenizer = cls(*init_inputs, **init_kwargs)
    except OSError:
        raise OSError(
            'Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.'
            )
    special_tokens_map_file = resolved_vocab_files.pop(
        'special_tokens_map_file', None)
    if special_tokens_map_file is not None:
        with open(special_tokens_map_file, encoding='utf-8'
            ) as special_tokens_map_handle:
            special_tokens_map = json.load(special_tokens_map_handle)
        for key, value in special_tokens_map.items():
            if isinstance(value, dict):
                value = AddedToken(**value)
            elif isinstance(value, list):
                value = [(AddedToken(**token) if isinstance(token, dict) else
                    token) for token in value]
            setattr(tokenizer, key, value)
    special_tokens = tokenizer.all_special_tokens
    if added_tokens_file is not None:
        with open(added_tokens_file, encoding='utf-8') as added_tokens_handle:
            added_tok_encoder = json.load(added_tokens_handle)
        added_tok_encoder_sorted = list(sorted(added_tok_encoder.items(),
            key=lambda x: x[1]))
        for token, index in added_tok_encoder_sorted:
            assert index == len(tokenizer
                ), f"Non-consecutive added token '{token}' found. Should have index {len(tokenizer)} but has index {index} in saved vocabulary."
            tokenizer.add_tokens(token, special_tokens=bool(token in
                special_tokens))
    added_tokens = tokenizer.sanitize_special_tokens()
    if added_tokens:
        logger.warning(
            'Special tokens have been added in the vocabulary, make sure the associated word embedding are fine-tuned or trained.'
            )
    return tokenizer
