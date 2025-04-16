def _load_empty_model(library_name: str, class_name: str,
    importable_classes: List[Any], pipelines: Any, is_pipeline_module: bool,
    name: str, torch_dtype: Union[str, torch.dtype], cached_folder: Union[
    str, os.PathLike], **kwargs):
    class_obj, _ = get_class_obj_and_candidates(library_name, class_name,
        importable_classes, pipelines, is_pipeline_module, component_name=
        name, cache_dir=cached_folder)
    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.
            __version__).base_version)
    else:
        transformers_version = 'N/A'
    is_transformers_model = is_transformers_available() and issubclass(
        class_obj, PreTrainedModel) and transformers_version >= version.parse(
        '4.20.0')
    diffusers_module = importlib.import_module(__name__.split('.')[0])
    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)
    model = None
    config_path = cached_folder
    user_agent = {'diffusers': __version__, 'file_type': 'model',
        'framework': 'pytorch'}
    if is_diffusers_model:
        config, unused_kwargs, commit_hash = class_obj.load_config(os.path.
            join(config_path, name), cache_dir=cached_folder,
            return_unused_kwargs=True, return_commit_hash=True,
            force_download=kwargs.pop('force_download', False),
            resume_download=kwargs.pop('resume_download', None), proxies=
            kwargs.pop('proxies', None), local_files_only=kwargs.pop(
            'local_files_only', False), token=kwargs.pop('token', None),
            revision=kwargs.pop('revision', None), subfolder=kwargs.pop(
            'subfolder', None), user_agent=user_agent)
        with accelerate.init_empty_weights():
            model = class_obj.from_config(config, **unused_kwargs)
    elif is_transformers_model:
        config_class = getattr(class_obj, 'config_class', None)
        if config_class is None:
            raise ValueError(
                '`config_class` cannot be None. Please double-check the model.'
                )
        config = config_class.from_pretrained(cached_folder, subfolder=name,
            force_download=kwargs.pop('force_download', False),
            resume_download=kwargs.pop('resume_download', None), proxies=
            kwargs.pop('proxies', None), local_files_only=kwargs.pop(
            'local_files_only', False), token=kwargs.pop('token', None),
            revision=kwargs.pop('revision', None), user_agent=user_agent)
        with accelerate.init_empty_weights():
            model = class_obj(config)
    if model is not None:
        model = model.to(dtype=torch_dtype)
    return model
