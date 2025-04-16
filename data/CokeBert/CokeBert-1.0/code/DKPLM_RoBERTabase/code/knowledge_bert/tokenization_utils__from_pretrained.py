@classmethod
def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs
    ):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    s3_models = list(cls.max_model_input_sizes.keys())
    vocab_files = {}
    init_configuration = {}
    if pretrained_model_name_or_path in s3_models:
        for file_id, map_list in cls.pretrained_vocab_files_map.items():
            vocab_files[file_id] = map_list[pretrained_model_name_or_path]
        if (cls.pretrained_init_configuration and 
            pretrained_model_name_or_path in cls.pretrained_init_configuration
            ):
            init_configuration = cls.pretrained_init_configuration[
                pretrained_model_name_or_path]
    else:
        logger.info(
            "Model name '{}' not found in model shortcut name list ({}). Assuming '{}' is a path or url to a directory containing tokenizer files."
            .format(pretrained_model_name_or_path, ', '.join(s3_models),
            pretrained_model_name_or_path))
        for file_id, file_name in cls.vocab_files_names.items():
            if os.path.isdir(pretrained_model_name_or_path):
                full_file_name = os.path.join(pretrained_model_name_or_path,
                    file_name)
                if not os.path.exists(full_file_name):
                    logger.info("Didn't find file {}. We won't load it.".
                        format(full_file_name))
                    full_file_name = None
            elif os.path.isfile(pretrained_model_name_or_path
                ) or is_remote_url(pretrained_model_name_or_path):
                full_file_name = pretrained_model_name_or_path
            else:
                full_file_name = hf_bucket_url(pretrained_model_name_or_path,
                    postfix=file_name)
            vocab_files[file_id] = full_file_name
        additional_files_names = {'added_tokens_file': ADDED_TOKENS_FILE,
            'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE,
            'tokenizer_config_file': TOKENIZER_CONFIG_FILE}
        saved_directory = pretrained_model_name_or_path
        if os.path.exists(saved_directory) and not os.path.isdir(
            saved_directory):
            saved_directory = os.path.dirname(saved_directory)
        for file_id, file_name in additional_files_names.items():
            full_file_name = os.path.join(saved_directory, file_name)
            if not os.path.exists(full_file_name):
                logger.info("Didn't find file {}. We won't load it.".format
                    (full_file_name))
                full_file_name = None
            vocab_files[file_id] = full_file_name
        if all(full_file_name is None for full_file_name in vocab_files.
            values()):
            raise EnvironmentError(
                "Model name '{}' was not found in tokenizers model name list ({}). We assumed '{}' was a path or url to a directory containing vocabulary files named {} but couldn't find such vocabulary files at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(s3_models),
                pretrained_model_name_or_path, list(cls.vocab_files_names.
                values())))
    try:
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None:
                resolved_vocab_files[file_id] = None
            else:
                resolved_vocab_files[file_id] = cached_path(file_path,
                    cache_dir=cache_dir, force_download=force_download,
                    proxies=proxies, resume_download=resume_download)
    except EnvironmentError:
        if pretrained_model_name_or_path in s3_models:
            msg = "Couldn't reach server at '{}' to download vocabulary files."
        else:
            msg = (
                "Model name '{}' was not found in tokenizers model name list ({}). We assumed '{}' was a path or url to a directory containing vocabulary files named {}, but couldn't find such vocabulary files at this path or url."
                .format(pretrained_model_name_or_path, ', '.join(s3_models),
                pretrained_model_name_or_path, list(cls.vocab_files_names.
                values())))
        raise EnvironmentError(msg)
    for file_id, file_path in vocab_files.items():
        if file_path == resolved_vocab_files[file_id]:
            logger.info('loading file {}'.format(file_path))
        else:
            logger.info('loading file {} from cache at {}'.format(file_path,
                resolved_vocab_files[file_id]))
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
    if pretrained_model_name_or_path in cls.max_model_input_sizes:
        max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
        if max_len is not None and isinstance(max_len, (int, float)):
            init_kwargs['max_len'] = min(init_kwargs.get('max_len', int(
                1000000000000.0)), max_len)
    added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
    special_tokens_map_file = resolved_vocab_files.pop(
        'special_tokens_map_file', None)
    for args_name, file_path in resolved_vocab_files.items():
        if args_name not in init_kwargs:
            init_kwargs[args_name] = file_path
    if special_tokens_map_file is not None:
        with open(special_tokens_map_file, encoding='utf-8'
            ) as special_tokens_map_handle:
            special_tokens_map = json.load(special_tokens_map_handle)
        for key, value in special_tokens_map.items():
            if key not in init_kwargs:
                init_kwargs[key] = value
    try:
        tokenizer = cls(*init_inputs, **init_kwargs)
    except OSError:
        OSError(
            'Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.'
            )
    tokenizer.init_inputs = init_inputs
    tokenizer.init_kwargs = init_kwargs
    if added_tokens_file is not None:
        with open(added_tokens_file, encoding='utf-8') as added_tokens_handle:
            added_tok_encoder = json.load(added_tokens_handle)
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        tokenizer.added_tokens_encoder.update(added_tok_encoder)
        tokenizer.added_tokens_decoder.update(added_tok_decoder)
    return tokenizer
