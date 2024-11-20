@classmethod
@validate_hf_hub_args
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
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
            custom_revision (`str`, *optional*):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. Defaults to the latest stable 🤗 Diffusers
                version.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

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
    cache_dir = kwargs.pop('cache_dir', None)
    resume_download = kwargs.pop('resume_download', None)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    from_flax = kwargs.pop('from_flax', False)
    torch_dtype = kwargs.pop('torch_dtype', None)
    custom_pipeline = kwargs.pop('custom_pipeline', None)
    custom_revision = kwargs.pop('custom_revision', None)
    provider = kwargs.pop('provider', None)
    sess_options = kwargs.pop('sess_options', None)
    device_map = kwargs.pop('device_map', None)
    max_memory = kwargs.pop('max_memory', None)
    offload_folder = kwargs.pop('offload_folder', None)
    offload_state_dict = kwargs.pop('offload_state_dict', False)
    low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage',
        _LOW_CPU_MEM_USAGE_DEFAULT)
    variant = kwargs.pop('variant', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    use_onnx = kwargs.pop('use_onnx', None)
    load_connected_pipeline = kwargs.pop('load_connected_pipeline', False)
    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            """Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: 
```
pip install accelerate
```
."""
            )
    if low_cpu_mem_usage is True and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`.'
            )
    if device_map is not None and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set `device_map=None`.'
            )
    if device_map is not None and not is_accelerate_available():
        raise NotImplementedError(
            'Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`.'
            )
    if device_map is not None and not isinstance(device_map, str):
        raise ValueError('`device_map` must be a string.')
    if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
        raise NotImplementedError(
            f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
            )
    if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
        if is_accelerate_version('<', '0.28.0'):
            raise NotImplementedError(
                'Device placement requires `accelerate` version `0.28.0` or later.'
                )
    if low_cpu_mem_usage is False and device_map is not None:
        raise ValueError(
            f'You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and dispatching. Please make sure to set `low_cpu_mem_usage=True`.'
            )
    if not os.path.isdir(pretrained_model_name_or_path):
        if pretrained_model_name_or_path.count('/') > 1:
            raise ValueError(
                f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}" is neither a valid local path nor a valid repo id. Please check the parameter.'
                )
        cached_folder = cls.download(pretrained_model_name_or_path,
            cache_dir=cache_dir, resume_download=resume_download,
            force_download=force_download, proxies=proxies,
            local_files_only=local_files_only, token=token, revision=
            revision, from_flax=from_flax, use_safetensors=use_safetensors,
            use_onnx=use_onnx, custom_pipeline=custom_pipeline,
            custom_revision=custom_revision, variant=variant,
            load_connected_pipeline=load_connected_pipeline, **kwargs)
    else:
        cached_folder = pretrained_model_name_or_path
    config_dict = cls.load_config(cached_folder)
    config_dict.pop('_ignore_files', None)
    model_variants = {}
    if variant is not None:
        for folder in os.listdir(cached_folder):
            folder_path = os.path.join(cached_folder, folder)
            is_folder = os.path.isdir(folder_path) and folder in config_dict
            variant_exists = is_folder and any(p.split('.')[1].startswith(
                variant) for p in os.listdir(folder_path))
            if variant_exists:
                model_variants[folder] = variant
    custom_class_name = None
    if os.path.isfile(os.path.join(cached_folder, f'{custom_pipeline}.py')):
        custom_pipeline = os.path.join(cached_folder, f'{custom_pipeline}.py')
    elif isinstance(config_dict['_class_name'], (list, tuple)
        ) and os.path.isfile(os.path.join(cached_folder,
        f"{config_dict['_class_name'][0]}.py")):
        custom_pipeline = os.path.join(cached_folder,
            f"{config_dict['_class_name'][0]}.py")
        custom_class_name = config_dict['_class_name'][1]
    pipeline_class = _get_pipeline_class(cls, config_dict,
        load_connected_pipeline=load_connected_pipeline, custom_pipeline=
        custom_pipeline, class_name=custom_class_name, cache_dir=cache_dir,
        revision=custom_revision)
    if device_map is not None and pipeline_class._load_connected_pipes:
        raise NotImplementedError(
            '`device_map` is not yet supported for connected pipelines.')
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
        init_dict and k not in pipeline_class._optional_components}
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
    from diffusers import pipelines
    final_device_map = None
    if device_map is not None:
        final_device_map = _get_final_device_map(device_map=device_map,
            pipeline_class=pipeline_class, passed_class_obj=
            passed_class_obj, init_dict=init_dict, library=library,
            max_memory=max_memory, torch_dtype=torch_dtype, cached_folder=
            cached_folder, force_download=force_download, resume_download=
            resume_download, proxies=proxies, local_files_only=
            local_files_only, token=token, revision=revision)
    current_device_map = None
    for name, (library_name, class_name) in logging.tqdm(init_dict.items(),
        desc='Loading pipeline components...'):
        if final_device_map is not None and len(final_device_map) > 0:
            component_device = final_device_map.get(name, None)
            if component_device is not None:
                current_device_map = {'': component_device}
            else:
                current_device_map = None
        class_name = class_name[4:] if class_name.startswith('Flax'
            ) else class_name
        is_pipeline_module = hasattr(pipelines, library_name)
        importable_classes = ALL_IMPORTABLE_CLASSES
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
                sess_options, device_map=current_device_map, max_memory=
                max_memory, offload_folder=offload_folder,
                offload_state_dict=offload_state_dict, model_variants=
                model_variants, name=name, from_flax=from_flax, variant=
                variant, low_cpu_mem_usage=low_cpu_mem_usage, cached_folder
                =cached_folder)
            logger.info(
                f'Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}.'
                )
        init_kwargs[name] = loaded_sub_model
    if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join
        (cached_folder, 'README.md')):
        modelcard = ModelCard.load(os.path.join(cached_folder, 'README.md'))
        connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[
            0] for prefix in CONNECTED_PIPES_KEYS}
        load_kwargs = {'cache_dir': cache_dir, 'resume_download':
            resume_download, 'force_download': force_download, 'proxies':
            proxies, 'local_files_only': local_files_only, 'token': token,
            'revision': revision, 'torch_dtype': torch_dtype,
            'custom_pipeline': custom_pipeline, 'custom_revision':
            custom_revision, 'provider': provider, 'sess_options':
            sess_options, 'device_map': device_map, 'max_memory':
            max_memory, 'offload_folder': offload_folder,
            'offload_state_dict': offload_state_dict, 'low_cpu_mem_usage':
            low_cpu_mem_usage, 'variant': variant, 'use_safetensors':
            use_safetensors}

        def get_connected_passed_kwargs(prefix):
            connected_passed_class_obj = {k.replace(f'{prefix}_', ''): w for
                k, w in passed_class_obj.items() if k.split('_')[0] == prefix}
            connected_passed_pipe_kwargs = {k.replace(f'{prefix}_', ''): w for
                k, w in passed_pipe_kwargs.items() if k.split('_')[0] == prefix
                }
            connected_passed_kwargs = {**connected_passed_class_obj, **
                connected_passed_pipe_kwargs}
            return connected_passed_kwargs
        connected_pipes = {prefix: DiffusionPipeline.from_pretrained(
            repo_id, **load_kwargs.copy(), **get_connected_passed_kwargs(
            prefix)) for prefix, repo_id in connected_pipes.items() if 
            repo_id is not None}
        for prefix, connected_pipe in connected_pipes.items():
            init_kwargs.update({'_'.join([prefix, name]): component for 
                name, component in connected_pipe.components.items()})
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
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    if device_map is not None:
        setattr(model, 'hf_device_map', final_device_map)
    return model
