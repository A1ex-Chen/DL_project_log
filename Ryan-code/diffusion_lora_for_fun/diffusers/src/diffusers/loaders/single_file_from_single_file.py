@classmethod
@validate_hf_hub_args
def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
    """
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
        format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            original_config_file (`str`, *optional*):
                The path to the original config file that was used to train the model. If not provided, the config file
                will be inferred from the checkpoint file.
            config (`str`, *optional*):
                Can be either:
                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing the pipeline
                      component configs in Diffusers format.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )

        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly.ckpt")

        >>> # Enable float16 and move to GPU
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipeline.to("cuda")
        ```

        """
    original_config_file = kwargs.pop('original_config_file', None)
    config = kwargs.pop('config', None)
    original_config = kwargs.pop('original_config', None)
    if original_config_file is not None:
        deprecation_message = (
            '`original_config_file` argument is deprecated and will be removed in future versions.please use the `original_config` argument instead.'
            )
        deprecate('original_config_file', '1.0.0', deprecation_message)
        original_config = original_config_file
    resume_download = kwargs.pop('resume_download', False)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    token = kwargs.pop('token', None)
    cache_dir = kwargs.pop('cache_dir', None)
    local_files_only = kwargs.pop('local_files_only', False)
    revision = kwargs.pop('revision', None)
    torch_dtype = kwargs.pop('torch_dtype', None)
    is_legacy_loading = False
    scaling_factor = kwargs.get('scaling_factor', None)
    if scaling_factor is not None:
        deprecation_message = (
            'Passing the `scaling_factor` argument to `from_single_file is deprecated and will be ignored in future versions.'
            )
        deprecate('scaling_factor', '1.0.0', deprecation_message)
    if original_config is not None:
        original_config = fetch_original_config(original_config,
            local_files_only=local_files_only)
    from ..pipelines.pipeline_utils import _get_pipeline_class
    pipeline_class = _get_pipeline_class(cls, config=None)
    checkpoint = load_single_file_checkpoint(pretrained_model_link_or_path,
        resume_download=resume_download, force_download=force_download,
        proxies=proxies, token=token, cache_dir=cache_dir, local_files_only
        =local_files_only, revision=revision)
    if config is None:
        config = fetch_diffusers_config(checkpoint)
        default_pretrained_model_config_name = config[
            'pretrained_model_name_or_path']
    else:
        default_pretrained_model_config_name = config
    if not os.path.isdir(default_pretrained_model_config_name):
        if default_pretrained_model_config_name.count('/') > 1:
            raise ValueError(
                f'The provided config "{config}" is neither a valid local path nor a valid repo id. Please check the parameter.'
                )
        try:
            cached_model_config_path = (
                _download_diffusers_model_config_from_hub(
                default_pretrained_model_config_name, cache_dir=cache_dir,
                revision=revision, proxies=proxies, force_download=
                force_download, resume_download=resume_download,
                local_files_only=local_files_only, token=token))
            config_dict = pipeline_class.load_config(cached_model_config_path)
        except LocalEntryNotFoundError:
            if original_config is None:
                logger.warning(
                    """`local_files_only` is True but no local configs were found for this checkpoint.
Attempting to download the necessary config files for this pipeline.
"""
                    )
                cached_model_config_path = (
                    _download_diffusers_model_config_from_hub(
                    default_pretrained_model_config_name, cache_dir=
                    cache_dir, revision=revision, proxies=proxies,
                    force_download=force_download, resume_download=
                    resume_download, local_files_only=False, token=token))
                config_dict = pipeline_class.load_config(
                    cached_model_config_path)
            else:
                logger.warning(
                    """Detected legacy `from_single_file` loading behavior. Attempting to create the pipeline based on inferred components.
This may lead to errors if the model components are not correctly inferred. 
To avoid this warning, please explicity pass the `config` argument to `from_single_file` with a path to a local diffusers model repo 
e.g. `from_single_file(<my model checkpoint path>, config=<path to local diffusers model repo>) 
or run `from_single_file` with `local_files_only=False` first to update the local cache directory with the necessary config files.
"""
                    )
                is_legacy_loading = True
                cached_model_config_path = None
                config_dict = _infer_pipeline_config_dict(pipeline_class)
                config_dict['_class_name'] = pipeline_class.__name__
    else:
        cached_model_config_path = default_pretrained_model_config_name
        config_dict = pipeline_class.load_config(cached_model_config_path)
    config_dict.pop('_ignore_files', None)
    expected_modules, optional_kwargs = pipeline_class._get_signature_keys(cls)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in
        kwargs}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in
        kwargs}
    init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict,
        **kwargs)
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in
        init_dict}
    init_kwargs = {**init_kwargs, **passed_pipe_kwargs}
    from diffusers import pipelines

    def load_module(name, value):
        if value[0] is None:
            return False
        if name in passed_class_obj and passed_class_obj[name] is None:
            return False
        if name in SINGLE_FILE_OPTIONAL_COMPONENTS:
            return False
        return True
    init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}
    for name, (library_name, class_name) in logging.tqdm(sorted(init_dict.
        items()), desc='Loading pipeline components...'):
        loaded_sub_model = None
        is_pipeline_module = hasattr(pipelines, library_name)
        if name in passed_class_obj:
            loaded_sub_model = passed_class_obj[name]
        else:
            try:
                loaded_sub_model = load_single_file_sub_model(library_name=
                    library_name, class_name=class_name, name=name,
                    checkpoint=checkpoint, is_pipeline_module=
                    is_pipeline_module, cached_model_config_path=
                    cached_model_config_path, pipelines=pipelines,
                    torch_dtype=torch_dtype, original_config=
                    original_config, local_files_only=local_files_only,
                    is_legacy_loading=is_legacy_loading, **kwargs)
            except SingleFileComponentError as e:
                raise SingleFileComponentError(
                    f"""{e.message}
Please load the component before passing it in as an argument to `from_single_file`.

{name} = {class_name}.from_pretrained('...')
pipe = {pipeline_class.__name__}.from_single_file(<checkpoint path>, {name}={name})

"""
                    )
        init_kwargs[name] = loaded_sub_model
    missing_modules = set(expected_modules) - set(init_kwargs.keys())
    passed_modules = list(passed_class_obj.keys())
    optional_modules = pipeline_class._optional_components
    if len(missing_modules) > 0 and missing_modules <= set(passed_modules +
        optional_modules):
        for module in missing_modules:
            init_kwargs[module] = passed_class_obj.get(module, None)
    elif len(missing_modules) > 0:
        passed_modules = set(list(init_kwargs.keys()) + list(
            passed_class_obj.keys())) - optional_kwargs
        raise ValueError(
            f'Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed.'
            )
    load_safety_checker = kwargs.pop('load_safety_checker', None)
    if load_safety_checker is not None:
        deprecation_message = (
            'Please pass instances of `StableDiffusionSafetyChecker` and `AutoImageProcessor`using the `safety_checker` and `feature_extractor` arguments in `from_single_file`'
            )
        deprecate('load_safety_checker', '1.0.0', deprecation_message)
        safety_checker_components = _legacy_load_safety_checker(
            local_files_only, torch_dtype)
        init_kwargs.update(safety_checker_components)
    pipe = pipeline_class(**init_kwargs)
    if torch_dtype is not None:
        pipe.to(dtype=torch_dtype)
    return pipe
