def convert_all_pt_checkpoints_to_tf(args_model_type, tf_dump_path,
    model_shortcut_names_or_path=None, config_shortcut_names_or_path=None,
    compare_with_pt_model=False, use_cached_models=False,
    remove_cached_files=False, only_convert_finetuned_models=False):
    if args_model_type is None:
        model_types = list(MODEL_CLASSES.keys())
    else:
        model_types = [args_model_type]
    for j, model_type in enumerate(model_types, start=1):
        print('=' * 100)
        print(' Converting model type {}/{}: {}'.format(j, len(model_types),
            model_type))
        print('=' * 100)
        if model_type not in MODEL_CLASSES:
            raise ValueError('Unrecognized model type {}, should be one of {}.'
                .format(model_type, list(MODEL_CLASSES.keys())))
        (config_class, model_class, pt_model_class, aws_model_maps,
            aws_config_map) = MODEL_CLASSES[model_type]
        if model_shortcut_names_or_path is None:
            model_shortcut_names_or_path = list(aws_model_maps.keys())
        if config_shortcut_names_or_path is None:
            config_shortcut_names_or_path = model_shortcut_names_or_path
        for i, (model_shortcut_name, config_shortcut_name) in enumerate(zip
            (model_shortcut_names_or_path, config_shortcut_names_or_path),
            start=1):
            print('-' * 100)
            if ('-squad' in model_shortcut_name or '-mrpc' in
                model_shortcut_name or '-mnli' in model_shortcut_name):
                if not only_convert_finetuned_models:
                    print('    Skipping finetuned checkpoint {}'.format(
                        model_shortcut_name))
                    continue
                model_type = model_shortcut_name
            elif only_convert_finetuned_models:
                print('    Skipping not finetuned checkpoint {}'.format(
                    model_shortcut_name))
                continue
            print('    Converting checkpoint {}/{}: {} - model_type {}'.
                format(i, len(aws_config_map), model_shortcut_name, model_type)
                )
            print('-' * 100)
            if config_shortcut_name in aws_config_map:
                config_file = cached_path(aws_config_map[
                    config_shortcut_name], force_download=not use_cached_models
                    )
            else:
                config_file = cached_path(config_shortcut_name,
                    force_download=not use_cached_models)
            if model_shortcut_name in aws_model_maps:
                model_file = cached_path(aws_model_maps[model_shortcut_name
                    ], force_download=not use_cached_models)
            else:
                model_file = cached_path(model_shortcut_name,
                    force_download=not use_cached_models)
            if os.path.isfile(model_shortcut_name):
                model_shortcut_name = 'converted_model'
            convert_pt_checkpoint_to_tf(model_type=model_type,
                pytorch_checkpoint_path=model_file, config_file=config_file,
                tf_dump_path=os.path.join(tf_dump_path, model_shortcut_name +
                '-tf_model.h5'), compare_with_pt_model=compare_with_pt_model)
            if remove_cached_files:
                os.remove(config_file)
                os.remove(model_file)
