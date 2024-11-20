@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a PyTorch diffusion pipeline from pre-trained pipeline weights.

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
                      [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                    This is an experimental feature and is likely to change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* of a custom pipeline hosted inside a model repo on
                      https://huggingface.co/. Valid repo ids have to be located under a user or organization name,
                      like `hf-internal-testing/diffusers-dummy-pipeline`.

                        <Tip>

                         It is required that the model repo has a file, called `pipeline.py` that defines the custom
                         pipeline.

                        </Tip>

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      https://github.com/huggingface/diffusers/tree/main/examples/community. Valid file names have to
                      match exactly the file name without `.py` located under the above link, *e.g.*
                      `clip_guided_stable_diffusion`.

                        <Tip>

                         Community pipelines are always loaded from the current `main` branch of GitHub.

                        </Tip>

                    - A path to a *directory* containing a custom pipeline, e.g., `./my_pipeline_directory/`.

                        <Tip>

                         It is required that the directory has a file, called `pipeline.py` that defines the custom
                         pipeline.

                        </Tip>

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
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
            custom_revision (`str`, *optional*, defaults to `"main"` when loading from the Hub and to local version of `diffusers` when loading from GitHub):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a diffusers version when loading a
                custom pipeline from GitHub.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            use_safetensors (`bool`, *optional* ):
                If set to `True`, the pipeline will be loaded from `safetensors` weights. If set to `None` (the
                default). The pipeline will load using `safetensors` if the safetensors weights are available *and* if
                `safetensors` is installed. If the to `False` the pipeline will *not* use `safetensors`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.

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
        >>> from diffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from diffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    resume_download = kwargs.pop('resume_download', False)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop('use_auth_token', None)
    revision = kwargs.pop('revision', None)
    from_flax = kwargs.pop('from_flax', False)
    torch_dtype = kwargs.pop('torch_dtype', None)
    custom_pipeline = kwargs.pop('custom_pipeline', None)
    custom_revision = kwargs.pop('custom_revision', None)
    provider = kwargs.pop('provider', None)
    sess_options = kwargs.pop('sess_options', None)
    device_map = kwargs.pop('device_map', None)
    low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage',
        _LOW_CPU_MEM_USAGE_DEFAULT)
    variant = kwargs.pop('variant', None)
    kwargs.pop('use_safetensors', None if is_safetensors_available() else False
        )
    if not os.path.isdir(pretrained_model_name_or_path):
        cached_folder = cls.download(pretrained_model_name_or_path,
            cache_dir=cache_dir, resume_download=resume_download,
            force_download=force_download, proxies=proxies,
            local_files_only=local_files_only, use_auth_token=
            use_auth_token, revision=revision, from_flax=from_flax,
            custom_pipeline=custom_pipeline, variant=variant)
    else:
        cached_folder = pretrained_model_name_or_path
    config_dict = cls.load_config(cached_folder)
    model_variants = {}
    if variant is not None:
        for folder in os.listdir(cached_folder):
            folder_path = os.path.join(cached_folder, folder)
            is_folder = os.path.isdir(folder_path) and folder in config_dict
            variant_exists = is_folder and any(path.split('.')[1] ==
                variant for path in os.listdir(folder_path))
            if variant_exists:
                model_variants[folder] = variant
    if custom_pipeline is not None:
        if custom_pipeline.endswith('.py'):
            path = Path(custom_pipeline)
            file_name = path.name
            custom_pipeline = path.parent.absolute()
        else:
            file_name = CUSTOM_PIPELINE_FILE_NAME
        pipeline_class = get_class_from_dynamic_module(custom_pipeline,
            module_file=file_name, cache_dir=cache_dir, revision=
            custom_revision)
    elif cls != DiffusionPipeline:
        pipeline_class = cls
    else:
        diffusers_module = importlib.import_module(cls.__module__.split('.')[0]
            )
        pipeline_class = getattr(diffusers_module, config_dict['_class_name'])
    if (pipeline_class.__name__ == 'StableDiffusionInpaintPipeline' and 
        version.parse(version.parse(config_dict['_diffusers_version']).
        base_version) <= version.parse('0.5.1')):
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy
        pipeline_class = StableDiffusionInpaintPipelineLegacy
        deprecation_message = (
            f"You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For better inpainting results, we strongly suggest using Stable Diffusion's official inpainting checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your checkpoint {pretrained_model_name_or_path} to the format of https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain the {{StableDiffusionInpaintPipelineLegacy}} class and will likely remove it in version 1.0.0."
            )
        deprecate('StableDiffusionInpaintPipelineLegacy', '1.0.0',
            deprecation_message, standard_warn=False)
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
    if (from_flax and 'safety_checker' in init_dict and 'safety_checker' not in
        passed_class_obj):
        raise NotImplementedError(
            'The safety checker cannot be automatically loaded when loading weights `from_flax`. Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker separately if you need it.'
            )
    if len(unused_kwargs) > 0:
        logger.warning(
            f'Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored.'
            )
    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            """Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: 
```
pip install accelerate
```
."""
            )
    if device_map is not None and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set `device_map=None`.'
            )
    if low_cpu_mem_usage is True and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`.'
            )
    if low_cpu_mem_usage is False and device_map is not None:
        raise ValueError(
            f'You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and dispatching. Please make sure to set `low_cpu_mem_usage=True`.'
            )
    from diffusers import pipelines
    for name, (library_name, class_name) in init_dict.items():
        if class_name.startswith('Flax'):
            class_name = class_name[4:]
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = (ALL_IMPORTABLE_CLASSES if is_pipeline_module else
            LOADABLE_CLASSES[library_name])
        loaded_sub_model = None
        if name in passed_class_obj:
            maybe_raise_or_warn(library_name, library, class_name,
                importable_classes, passed_class_obj, name, is_pipeline_module)
            loaded_sub_model = passed_class_obj[name]
        else:
            loaded_sub_model = load_sub_model(library_name=library_name,
                class_name=class_name, importable_classes=
                importable_classes, pipelines=pipelines, is_pipeline_module
                =is_pipeline_module, pipeline_class=pipeline_class,
                torch_dtype=torch_dtype, provider=provider, sess_options=
                sess_options, device_map=device_map, model_variants=
                model_variants, name=name, from_flax=from_flax, variant=
                variant, low_cpu_mem_usage=low_cpu_mem_usage, cached_folder
                =cached_folder)
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
    model = pipeline_class(**init_kwargs)
    return_cached_folder = kwargs.pop('return_cached_folder', False)
    if return_cached_folder:
        message = f"""Passing `return_cached_folder=True` is deprecated and will be removed in `diffusers=0.17.0`. Please do the following instead: 
 1. Load the cached_folder via `cached_folder={cls}.download({pretrained_model_name_or_path})`. 
 2. Load the pipeline by loading from the cached folder: `pipeline={cls}.from_pretrained(cached_folder)`."""
        deprecate('return_cached_folder', '0.17.0', message, take_from=kwargs)
        return model, cached_folder
    return model
