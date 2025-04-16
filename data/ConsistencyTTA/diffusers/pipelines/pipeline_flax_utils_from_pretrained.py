@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a Flax diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                      https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                      `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~FlaxDiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            dtype (`str` or `jnp.dtype`, *optional*):
                Override the default `jnp.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models), *e.g.* `"runwayml/stable-diffusion-v1-5"`

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

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
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    use_auth_token = kwargs.pop('use_auth_token', None)
    revision = kwargs.pop('revision', None)
    from_pt = kwargs.pop('from_pt', False)
    dtype = kwargs.pop('dtype', None)
    if not os.path.isdir(pretrained_model_name_or_path):
        config_dict = cls.load_config(pretrained_model_name_or_path,
            cache_dir=cache_dir, resume_download=resume_download, proxies=
            proxies, local_files_only=local_files_only, use_auth_token=
            use_auth_token, revision=revision)
        folder_names = [k for k in config_dict.keys() if not k.startswith('_')]
        allow_patterns = [os.path.join(k, '*') for k in folder_names]
        allow_patterns += [FLAX_WEIGHTS_NAME, SCHEDULER_CONFIG_NAME,
            CONFIG_NAME, cls.config_name]
        ignore_patterns = '*.bin' if not from_pt else []
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
            proxies, local_files_only=local_files_only, use_auth_token=
            use_auth_token, revision=revision, allow_patterns=
            allow_patterns, ignore_patterns=ignore_patterns, user_agent=
            user_agent)
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
    init_dict, _, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)
    init_kwargs = {}
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
                    from_pt=from_pt, dtype=dtype)
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
