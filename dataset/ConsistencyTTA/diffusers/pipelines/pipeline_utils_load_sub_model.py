def load_sub_model(library_name: str, class_name: str, importable_classes:
    List[Any], pipelines: Any, is_pipeline_module: bool, pipeline_class:
    Any, torch_dtype: torch.dtype, provider: Any, sess_options: Any,
    device_map: Optional[Union[Dict[str, torch.device], str]],
    model_variants: Dict[str, str], name: str, from_flax: bool, variant:
    str, low_cpu_mem_usage: bool, cached_folder: Union[str, os.PathLike]):
    """Helper method to load the module `name` from `library_name` and `class_name`"""
    class_obj, class_candidates = get_class_obj_and_candidates(library_name,
        class_name, importable_classes, pipelines, is_pipeline_module)
    load_method_name = None
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj,
            class_candidate):
            load_method_name = importable_classes[class_name][1]
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER
            ) or none_module.startswith(TRANSFORMERS_DUMMY_MODULES_FOLDER)
        if is_dummy_path and 'dummy' in none_module:
            class_obj()
        raise ValueError(
            f'The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}.'
            )
    load_method = getattr(class_obj, load_method_name)
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs['torch_dtype'] = torch_dtype
    if issubclass(class_obj, diffusers.OnnxRuntimeModel):
        loading_kwargs['provider'] = provider
        loading_kwargs['sess_options'] = sess_options
    is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)
    if is_transformers_available():
        transformers_version = version.parse(version.parse(transformers.
            __version__).base_version)
    else:
        transformers_version = 'N/A'
    is_transformers_model = is_transformers_available() and issubclass(
        class_obj, PreTrainedModel) and transformers_version >= version.parse(
        '4.20.0')
    if is_diffusers_model or is_transformers_model:
        loading_kwargs['device_map'] = device_map
        loading_kwargs['variant'] = model_variants.pop(name, None)
        if from_flax:
            loading_kwargs['from_flax'] = True
        if is_transformers_model and loading_kwargs['variant'
            ] is not None and transformers_version < version.parse('4.27.0'):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
                )
        elif is_transformers_model and loading_kwargs['variant'] is None:
            loading_kwargs.pop('variant')
        if not (from_flax and is_transformers_model):
            loading_kwargs['low_cpu_mem_usage'] = low_cpu_mem_usage
        else:
            loading_kwargs['low_cpu_mem_usage'] = False
    if os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(os.path.join(cached_folder, name),
            **loading_kwargs)
    else:
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)
    return loaded_sub_model
