@classmethod
@validate_hf_hub_args
def download(cls, pretrained_model_name, **kwargs) ->Union[str, os.PathLike]:
    """
        Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.

        Parameters:
            pretrained_model_name (`str` or `os.PathLike`, *optional*):
                A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                hosted on the Hub.
            custom_pipeline (`str`, *optional*):
                Can be either:

                    - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
                      pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
                      the custom pipeline.

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current `main` branch of GitHub.

                    - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                For more information on how to load and create custom pipelines, take a look at [How to contribute a
                community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).

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
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `False`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom pipelines and components defined on the Hub in their own files. This
                option should only be set to `True` for repositories you trust and in which you have read the code, as
                it will execute code present on the Hub on your local machine.

        Returns:
            `os.PathLike`:
                A path to the downloaded pipeline.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`.

        </Tip>

        """
    cache_dir = kwargs.pop('cache_dir', None)
    resume_download = kwargs.pop('resume_download', None)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    from_flax = kwargs.pop('from_flax', False)
    custom_pipeline = kwargs.pop('custom_pipeline', None)
    custom_revision = kwargs.pop('custom_revision', None)
    variant = kwargs.pop('variant', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    use_onnx = kwargs.pop('use_onnx', None)
    load_connected_pipeline = kwargs.pop('load_connected_pipeline', False)
    trust_remote_code = kwargs.pop('trust_remote_code', False)
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True
    allow_patterns = None
    ignore_patterns = None
    model_info_call_error: Optional[Exception] = None
    if not local_files_only:
        try:
            info = model_info(pretrained_model_name, token=token, revision=
                revision)
        except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError
            ) as e:
            logger.warning(
                f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache."
                )
            local_files_only = True
            model_info_call_error = e
    if not local_files_only:
        config_file = hf_hub_download(pretrained_model_name, cls.
            config_name, cache_dir=cache_dir, revision=revision, proxies=
            proxies, force_download=force_download, resume_download=
            resume_download, token=token)
        config_dict = cls._dict_from_json_file(config_file)
        ignore_filenames = config_dict.pop('_ignore_files', [])
        folder_names = [k for k, v in config_dict.items() if isinstance(v,
            list) and k != '_class_name']
        filenames = {sibling.rfilename for sibling in info.siblings}
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant)
        diffusers_module = importlib.import_module(__name__.split('.')[0])
        pipelines = getattr(diffusers_module, 'pipelines')
        custom_components = {}
        for component in folder_names:
            module_candidate = config_dict[component][0]
            if module_candidate is None or not isinstance(module_candidate, str
                ):
                continue
            candidate_file = f'{component}/{module_candidate}.py'
            if candidate_file in filenames:
                custom_components[component] = module_candidate
            elif module_candidate not in LOADABLE_CLASSES and not hasattr(
                pipelines, module_candidate):
                raise ValueError(
                    f"{candidate_file} as defined in `model_index.json` does not exist in {pretrained_model_name} and is not a module in 'diffusers/pipelines'."
                    )
        if len(variant_filenames) == 0 and variant is not None:
            deprecation_message = (
                f'You are trying to load the model files of the `variant={variant}`, but no such modeling files are available.The default model files: {model_filenames} will be loaded instead. Make sure to not load from `variant={variant}`if such variant modeling files are not available. Doing so will lead to an error in v0.24.0 as defaulting to non-variantmodeling files is deprecated.'
                )
            deprecate('no variant default', '0.24.0', deprecation_message,
                standard_warn=False)
        model_filenames = set(model_filenames) - set(ignore_filenames)
        variant_filenames = set(variant_filenames) - set(ignore_filenames)
        if revision in DEPRECATED_REVISION_ARGS and version.parse(version.
            parse(__version__).base_version) >= version.parse('0.22.0'):
            warn_deprecated_model_variant(pretrained_model_name, token,
                variant, revision, model_filenames)
        model_folder_names = {os.path.split(f)[0] for f in model_filenames if
            os.path.split(f)[0] in folder_names}
        custom_class_name = None
        if custom_pipeline is None and isinstance(config_dict['_class_name'
            ], (list, tuple)):
            custom_pipeline = config_dict['_class_name'][0]
            custom_class_name = config_dict['_class_name'][1]
        allow_patterns = list(model_filenames)
        allow_patterns += [f'{k}/*' for k in folder_names if k not in
            model_folder_names]
        allow_patterns += [f'{k}/{f}.py' for k, f in custom_components.items()]
        allow_patterns += [f'{custom_pipeline}.py'
            ] if f'{custom_pipeline}.py' in filenames else []
        allow_patterns += [os.path.join(k, 'config.json') for k in
            model_folder_names]
        allow_patterns += [SCHEDULER_CONFIG_NAME, CONFIG_NAME, cls.
            config_name, CUSTOM_PIPELINE_FILE_NAME]
        load_pipe_from_hub = (custom_pipeline is not None and 
            f'{custom_pipeline}.py' in filenames)
        load_components_from_hub = len(custom_components) > 0
        if load_pipe_from_hub and not trust_remote_code:
            raise ValueError(
                f"""The repository for {pretrained_model_name} contains custom code in {custom_pipeline}.py which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/{pretrained_model_name}/blob/main/{custom_pipeline}.py.
Please pass the argument `trust_remote_code=True` to allow custom code to be run."""
                )
        if load_components_from_hub and not trust_remote_code:
            raise ValueError(
                f"""The repository for {pretrained_model_name} contains custom code in {'.py, '.join([os.path.join(k, v) for k, v in custom_components.items()])} which must be executed to correctly load the model. You can inspect the repository content at {', '.join([f'https://hf.co/{pretrained_model_name}/{k}/{v}.py' for k, v in custom_components.items()])}.
Please pass the argument `trust_remote_code=True` to allow custom code to be run."""
                )
        pipeline_class = _get_pipeline_class(cls, config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline, repo_id=pretrained_model_name if
            load_pipe_from_hub else None, hub_revision=revision, class_name
            =custom_class_name, cache_dir=cache_dir, revision=custom_revision)
        expected_components, _ = cls._get_signature_keys(pipeline_class)
        passed_components = [k for k in expected_components if k in kwargs]
        if (use_safetensors and not allow_pickle and not
            is_safetensors_compatible(model_filenames, variant=variant,
            passed_components=passed_components)):
            raise EnvironmentError(
                f'Could not find the necessary `safetensors` weights in {model_filenames} (variant={variant})'
                )
        if from_flax:
            ignore_patterns = ['*.bin', '*.safetensors', '*.onnx', '*.pb']
        elif use_safetensors and is_safetensors_compatible(model_filenames,
            variant=variant, passed_components=passed_components):
            ignore_patterns = ['*.bin', '*.msgpack']
            use_onnx = (use_onnx if use_onnx is not None else
                pipeline_class._is_onnx)
            if not use_onnx:
                ignore_patterns += ['*.onnx', '*.pb']
            safetensors_variant_filenames = {f for f in variant_filenames if
                f.endswith('.safetensors')}
            safetensors_model_filenames = {f for f in model_filenames if f.
                endswith('.safetensors')}
            if (len(safetensors_variant_filenames) > 0 and 
                safetensors_model_filenames != safetensors_variant_filenames):
                logger.warning(
                    f"""
A mixture of {variant} and non-{variant} filenames will be loaded.
Loaded {variant} filenames:
[{', '.join(safetensors_variant_filenames)}]
Loaded non-{variant} filenames:
[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}
If this behavior is not expected, please check your folder structure."""
                    )
        else:
            ignore_patterns = ['*.safetensors', '*.msgpack']
            use_onnx = (use_onnx if use_onnx is not None else
                pipeline_class._is_onnx)
            if not use_onnx:
                ignore_patterns += ['*.onnx', '*.pb']
            bin_variant_filenames = {f for f in variant_filenames if f.
                endswith('.bin')}
            bin_model_filenames = {f for f in model_filenames if f.endswith
                ('.bin')}
            if len(bin_variant_filenames
                ) > 0 and bin_model_filenames != bin_variant_filenames:
                logger.warning(
                    f"""
A mixture of {variant} and non-{variant} filenames will be loaded.
Loaded {variant} filenames:
[{', '.join(bin_variant_filenames)}]
Loaded non-{variant} filenames:
[{', '.join(bin_model_filenames - bin_variant_filenames)}
If this behavior is not expected, please check your folder structure."""
                    )
        allow_patterns = [p for p in allow_patterns if not (len(p.split('/'
            )) == 2 and p.split('/')[0] in passed_components)]
        if pipeline_class._load_connected_pipes:
            allow_patterns.append('README.md')
        ignore_patterns = ignore_patterns + [f'{i}.index.*json' for i in
            ignore_patterns]
        re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in
            ignore_patterns]
        re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in
            allow_patterns]
        expected_files = [f for f in filenames if not any(p.match(f) for p in
            re_ignore_pattern)]
        expected_files = [f for f in expected_files if any(p.match(f) for p in
            re_allow_pattern)]
        snapshot_folder = Path(config_file).parent
        pipeline_is_cached = all((snapshot_folder / f).is_file() for f in
            expected_files)
        if pipeline_is_cached and not force_download:
            return snapshot_folder
    user_agent = {'pipeline_class': cls.__name__}
    if custom_pipeline is not None and not custom_pipeline.endswith('.py'):
        user_agent['custom_pipeline'] = custom_pipeline
    try:
        cached_folder = snapshot_download(pretrained_model_name, cache_dir=
            cache_dir, resume_download=resume_download, proxies=proxies,
            local_files_only=local_files_only, token=token, revision=
            revision, allow_patterns=allow_patterns, ignore_patterns=
            ignore_patterns, user_agent=user_agent)
        cls_name = cls.load_config(os.path.join(cached_folder,
            'model_index.json')).get('_class_name', None)
        cls_name = cls_name[4:] if isinstance(cls_name, str
            ) and cls_name.startswith('Flax') else cls_name
        diffusers_module = importlib.import_module(__name__.split('.')[0])
        pipeline_class = getattr(diffusers_module, cls_name, None
            ) if isinstance(cls_name, str) else None
        if pipeline_class is not None and pipeline_class._load_connected_pipes:
            modelcard = ModelCard.load(os.path.join(cached_folder, 'README.md')
                )
            connected_pipes = sum([getattr(modelcard.data, k, []) for k in
                CONNECTED_PIPES_KEYS], [])
            for connected_pipe_repo_id in connected_pipes:
                download_kwargs = {'cache_dir': cache_dir,
                    'resume_download': resume_download, 'force_download':
                    force_download, 'proxies': proxies, 'local_files_only':
                    local_files_only, 'token': token, 'variant': variant,
                    'use_safetensors': use_safetensors}
                DiffusionPipeline.download(connected_pipe_repo_id, **
                    download_kwargs)
        return cached_folder
    except FileNotFoundError:
        if model_info_call_error is None:
            raise
        else:
            raise EnvironmentError(
                f'Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.'
                ) from model_info_call_error
