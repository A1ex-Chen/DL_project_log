def load_single_file_sub_model(library_name, class_name, name, checkpoint,
    pipelines, is_pipeline_module, cached_model_config_path,
    original_config=None, local_files_only=False, torch_dtype=None,
    is_legacy_loading=False, **kwargs):
    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.
            __version__).base_version)
    else:
        transformers_version = 'N/A'
    is_transformers_model = is_transformers_available() and issubclass(
        class_obj, PreTrainedModel) and transformers_version >= version.parse(
        '4.20.0')
    is_tokenizer = is_transformers_available() and issubclass(class_obj,
        PreTrainedTokenizer) and transformers_version >= version.parse('4.20.0'
        )
    diffusers_module = importlib.import_module(__name__.split('.')[0])
    is_diffusers_single_file_model = issubclass(class_obj, diffusers_module
        .FromOriginalModelMixin)
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    is_diffusers_scheduler = issubclass(class_obj, diffusers_module.
        SchedulerMixin)
    if is_diffusers_single_file_model:
        load_method = getattr(class_obj, 'from_single_file')
        if original_config:
            cached_model_config_path = None
        loaded_sub_model = load_method(pretrained_model_link_or_path_or_dict
            =checkpoint, original_config=original_config, config=
            cached_model_config_path, subfolder=name, torch_dtype=
            torch_dtype, local_files_only=local_files_only, **kwargs)
    elif is_transformers_model and is_clip_model_in_single_file(class_obj,
        checkpoint):
        loaded_sub_model = create_diffusers_clip_model_from_ldm(class_obj,
            checkpoint=checkpoint, config=cached_model_config_path,
            subfolder=name, torch_dtype=torch_dtype, local_files_only=
            local_files_only, is_legacy_loading=is_legacy_loading)
    elif is_tokenizer and is_legacy_loading:
        loaded_sub_model = _legacy_load_clip_tokenizer(class_obj,
            checkpoint=checkpoint, config=cached_model_config_path,
            local_files_only=local_files_only)
    elif is_diffusers_scheduler and is_legacy_loading:
        loaded_sub_model = _legacy_load_scheduler(class_obj, checkpoint=
            checkpoint, component_name=name, original_config=
            original_config, **kwargs)
    else:
        if not hasattr(class_obj, 'from_pretrained'):
            raise ValueError(
                f'The component {class_obj.__name__} cannot be loaded as it does not seem to have a supported loading method.'
                )
        loading_kwargs = {}
        loading_kwargs.update({'pretrained_model_name_or_path':
            cached_model_config_path, 'subfolder': name, 'local_files_only':
            local_files_only})
        if issubclass(class_obj, torch.nn.Module):
            loading_kwargs.update({'torch_dtype': torch_dtype})
        if is_diffusers_model or is_transformers_model:
            if not _is_model_weights_in_cached_folder(cached_model_config_path,
                name):
                raise SingleFileComponentError(
                    f'Failed to load {class_name}. Weights for this component appear to be missing in the checkpoint.'
                    )
        load_method = getattr(class_obj, 'from_pretrained')
        loaded_sub_model = load_method(**loading_kwargs)
    return loaded_sub_model
