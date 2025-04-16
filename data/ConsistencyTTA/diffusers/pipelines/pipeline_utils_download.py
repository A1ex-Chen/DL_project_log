@classmethod
def download(cls, pretrained_model_name, **kwargs) ->Union[str, os.PathLike]:
    """
        Download and cache a PyTorch diffusion pipeline from pre-trained pipeline weights.

        Parameters:
             pretrained_model_name (`str` or `os.PathLike`, *optional*):
                Should be a string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                `CompVis/ldm-text2im-large-256`.
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
            custom_revision (`str`, *optional*, defaults to `"main"` when loading from the Hub and to local version of
            `diffusers` when loading from GitHub):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a diffusers version when loading a
                custom pipeline from GitHub.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models)

        </Tip>

        """
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    resume_download = kwargs.pop('resume_download', False)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop('use_auth_token', None)
    revision = kwargs.pop('revision', None)
    from_flax = kwargs.pop('from_flax', False)
    custom_pipeline = kwargs.pop('custom_pipeline', None)
    variant = kwargs.pop('variant', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    if use_safetensors and not is_safetensors_available():
        raise ValueError(
            '`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors'
            )
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = is_safetensors_available()
        allow_pickle = True
    pipeline_is_cached = False
    allow_patterns = None
    ignore_patterns = None
    if not local_files_only:
        config_file = hf_hub_download(pretrained_model_name, cls.
            config_name, cache_dir=cache_dir, revision=revision, proxies=
            proxies, force_download=force_download, resume_download=
            resume_download, use_auth_token=use_auth_token)
        info = model_info(pretrained_model_name, use_auth_token=
            use_auth_token, revision=revision)
        config_dict = cls._dict_from_json_file(config_file)
        folder_names = [k for k, v in config_dict.items() if isinstance(v,
            list)]
        filenames = {sibling.rfilename for sibling in info.siblings}
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant=variant)
        if revision in DEPRECATED_REVISION_ARGS and version.parse(version.
            parse(__version__).base_version) >= version.parse('0.17.0'):
            warn_deprecated_model_variant(pretrained_model_name,
                use_auth_token, variant, revision, model_filenames)
        model_folder_names = {os.path.split(f)[0] for f in model_filenames}
        allow_patterns = list(model_filenames)
        allow_patterns += [os.path.join(k, '*') for k in folder_names if k
             not in model_folder_names]
        allow_patterns += [os.path.join(k, '*.json') for k in
            model_folder_names]
        allow_patterns += [SCHEDULER_CONFIG_NAME, CONFIG_NAME, cls.
            config_name, CUSTOM_PIPELINE_FILE_NAME]
        if (use_safetensors and not allow_pickle and not
            is_safetensors_compatible(model_filenames, variant=variant)):
            raise EnvironmentError(
                f'Could not found the necessary `safetensors` weights in {model_filenames} (variant={variant})'
                )
        if from_flax:
            ignore_patterns = ['*.bin', '*.safetensors', '*.onnx', '*.pb']
        elif use_safetensors and is_safetensors_compatible(model_filenames,
            variant=variant):
            ignore_patterns = ['*.bin', '*.msgpack']
            safetensors_variant_filenames = {f for f in variant_filenames if
                f.endswith('.safetensors')}
            safetensors_model_filenames = {f for f in model_filenames if f.
                endswith('.safetensors')}
            if (len(safetensors_variant_filenames) > 0 and 
                safetensors_model_filenames != safetensors_variant_filenames):
                logger.warn(
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
            bin_variant_filenames = {f for f in variant_filenames if f.
                endswith('.bin')}
            bin_model_filenames = {f for f in model_filenames if f.endswith
                ('.bin')}
            if len(bin_variant_filenames
                ) > 0 and bin_model_filenames != bin_variant_filenames:
                logger.warn(
                    f"""
A mixture of {variant} and non-{variant} filenames will be loaded.
Loaded {variant} filenames:
[{', '.join(bin_variant_filenames)}]
Loaded non-{variant} filenames:
[{', '.join(bin_model_filenames - bin_variant_filenames)}
If this behavior is not expected, please check your folder structure."""
                    )
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
        if pipeline_is_cached:
            return snapshot_folder
    user_agent = {'pipeline_class': cls.__name__}
    if custom_pipeline is not None and not custom_pipeline.endswith('.py'):
        user_agent['custom_pipeline'] = custom_pipeline
    cached_folder = snapshot_download(pretrained_model_name, cache_dir=
        cache_dir, resume_download=resume_download, proxies=proxies,
        local_files_only=local_files_only, use_auth_token=use_auth_token,
        revision=revision, allow_patterns=allow_patterns, ignore_patterns=
        ignore_patterns, user_agent=user_agent)
    return cached_folder
