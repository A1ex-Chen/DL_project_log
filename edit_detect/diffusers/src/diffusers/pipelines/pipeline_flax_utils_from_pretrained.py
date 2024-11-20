@classmethod
@validate_hf_hub_args
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a Flax-based diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()) by default and dropout modules are deactivated.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of FlaxUNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `runwayml/stable-diffusion-v1-5`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      using [`~FlaxDiffusionPipeline.save_pretrained`].
            dtype (`str` or `jnp.dtype`, *optional*):
                Override the default `jnp.dtype` and load the model under this dtype. If `"auto"`, the dtype is
                automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components) of the specific pipeline
                class. The overwritten components are passed directly to the pipelines `__init__` method.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from diffusers import FlaxDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> # Requires to be logged in to Hugging Face hub,
        >>> # see more in [the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline, params = FlaxDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5",
        ...     revision="bf16",
        ...     dtype=jnp.bfloat16,
        ... )

        >>> # Download pipeline, but use a different scheduler
        >>> from diffusers import FlaxDPMSolverMultistepScheduler

        >>> model_id = "runwayml/stable-diffusion-v1-5"
        >>> dpmpp, dpmpp_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
        ...     model_id,
        ...     subfolder="scheduler",
        ... )

        >>> dpm_pipe, dpm_params = FlaxStableDiffusionPipeline.from_pretrained(
        ...     model_id, revision="bf16", dtype=jnp.bfloat16, scheduler=dpmpp
        ... )
        >>> dpm_params["scheduler"] = dpmpp_state
        ```
        """
    cache_dir = kwargs.pop('cache_dir', None)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    from_pt = kwargs.pop('from_pt', False)
    use_memory_efficient_attention = kwargs.pop(
        'use_memory_efficient_attention', False)
    split_head_dim = kwargs.pop('split_head_dim', False)
    dtype = kwargs.pop('dtype', None)
    if not os.path.isdir(pretrained_model_name_or_path):
        config_dict = cls.load_config(pretrained_model_name_or_path,
            cache_dir=cache_dir, resume_download=resume_download, proxies=
            proxies, local_files_only=local_files_only, token=token,
            revision=revision)
        folder_names = [k for k in config_dict.keys() if not k.startswith('_')]
        allow_patterns = [os.path.join(k, '*') for k in folder_names]
        allow_patterns += [FLAX_WEIGHTS_NAME, SCHEDULER_CONFIG_NAME,
            CONFIG_NAME, cls.config_name]
        ignore_patterns = ['*.bin', '*.safetensors'] if not from_pt else []
        ignore_patterns += ['*.onnx', '*.onnx_data', '*.xml', '*.pb']
        if cls != FlaxDiffusionPipeline:
            requested_pipeline_class = cls.__name__
        else:
            requested_pipeline_class = config_dict.get('_class_name', cls.
                __name__)
            requested_pipeline_class = (requested_pipeline_class if
                requested_pipeline_class.startswith('Flax') else 'Flax' +
                requested_pipeline_class)
        user_agent = {'pipeline_class': requested_pipeline_class}
        user_agent = http_user_agent(user_agent)
        cached_folder = snapshot_download(pretrained_model_name_or_path,
            cache_dir=cache_dir, resume_download=resume_download, proxies=
            proxies, local_files_only=local_files_only, token=token,
            revision=revision, allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns, user_agent=user_agent)
    else:
        cached_folder = pretrained_model_name_or_path
    config_dict = cls.load_config(cached_folder)
    if cls != FlaxDiffusionPipeline:
        pipeline_class = cls
    else:
        diffusers_module = importlib.import_module(cls.__module__.split('.')[0]
            )
        class_name = config_dict['_class_name'] if config_dict['_class_name'
            ].startswith('Flax') else 'Flax' + config_dict['_class_name']
        pipeline_class = getattr(diffusers_module, class_name)
    expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
    passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in
        kwargs}
    passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in
        kwargs}
    init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict,
        **kwargs)
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in
        init_dict}
    init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

    def load_module(name, value):
        if value[0] is None:
            return False
        if name in passed_class_obj and passed_class_obj[name] is None:
            return False
        return True
    init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}
    if len(unused_kwargs) > 0:
        logger.warning(
            f'Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored.'
            )
    params = {}
    from diffusers import pipelines
    for name, (library_name, class_name) in init_dict.items():
        if class_name is None:
            init_kwargs[name] = None
            continue
        is_pipeline_module = hasattr(pipelines, library_name)
        loaded_sub_model = None
        sub_model_should_be_defined = True
        if name in passed_class_obj:
            if not is_pipeline_module:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in
                    importable_classes.keys()}
                expected_class_obj = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj,
                        class_candidate):
                        expected_class_obj = class_candidate
                if not issubclass(passed_class_obj[name].__class__,
                    expected_class_obj):
                    raise ValueError(
                        f'{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be {expected_class_obj}'
                        )
            elif passed_class_obj[name] is None:
                logger.warning(
                    f'You have passed `None` for {name} to disable its functionality in {pipeline_class}. Note that this might lead to problems when using {pipeline_class} and is not recommended.'
                    )
                sub_model_should_be_defined = False
            else:
                logger.warning(
                    f'You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it has the correct type'
                    )
            loaded_sub_model = passed_class_obj[name]
        elif is_pipeline_module:
            pipeline_module = getattr(pipelines, library_name)
            class_obj = import_flax_or_no_model(pipeline_module, class_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            class_candidates = {c: class_obj for c in importable_classes.keys()
                }
        else:
            library = importlib.import_module(library_name)
            class_obj = import_flax_or_no_model(library, class_name)
            importable_classes = LOADABLE_CLASSES[library_name]
            class_candidates = {c: getattr(library, c, None) for c in
                importable_classes.keys()}
        if loaded_sub_model is None and sub_model_should_be_defined:
            load_method_name = None
            for class_name, class_candidate in class_candidates.items():
                if class_candidate is not None and issubclass(class_obj,
                    class_candidate):
                    load_method_name = importable_classes[class_name][1]
            load_method = getattr(class_obj, load_method_name)
            if os.path.isdir(os.path.join(cached_folder, name)):
                loadable_folder = os.path.join(cached_folder, name)
            else:
                loaded_sub_model = cached_folder
            if issubclass(class_obj, FlaxModelMixin):
                loaded_sub_model, loaded_params = load_method(loadable_folder,
                    from_pt=from_pt, use_memory_efficient_attention=
                    use_memory_efficient_attention, split_head_dim=
                    split_head_dim, dtype=dtype)
                params[name] = loaded_params
            elif is_transformers_available() and issubclass(class_obj,
                FlaxPreTrainedModel):
                if from_pt:
                    loaded_sub_model = load_method(loadable_folder, from_pt
                        =from_pt)
                    loaded_params = loaded_sub_model.params
                    del loaded_sub_model._params
                else:
                    loaded_sub_model, loaded_params = load_method(
                        loadable_folder, _do_init=False)
                params[name] = loaded_params
            elif issubclass(class_obj, FlaxSchedulerMixin):
                loaded_sub_model, scheduler_state = load_method(loadable_folder
                    )
                params[name] = scheduler_state
            else:
                loaded_sub_model = load_method(loadable_folder)
        init_kwargs[name] = loaded_sub_model
    missing_modules = set(expected_modules) - set(init_kwargs.keys())
    passed_modules = list(passed_class_obj.keys())
    if len(missing_modules) > 0 and missing_modules <= set(passed_modules):
        for module in missing_modules:
            init_kwargs[module] = passed_class_obj.get(module, None)
    elif len(missing_modules) > 0:
        passed_modules = set(list(init_kwargs.keys()) + list(
            passed_class_obj.keys())) - optional_kwargs
        raise ValueError(
            f'Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed.'
            )
    model = pipeline_class(**init_kwargs, dtype=dtype)
    return model, params
