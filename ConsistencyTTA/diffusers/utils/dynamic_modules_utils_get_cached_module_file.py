def get_cached_module_file(pretrained_model_name_or_path: Union[str, os.
    PathLike], module_file: str, cache_dir: Optional[Union[str, os.PathLike
    ]]=None, force_download: bool=False, resume_download: bool=False,
    proxies: Optional[Dict[str, str]]=None, use_auth_token: Optional[Union[
    bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=
    False):
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    You may pass a token in `use_auth_token` if you are not logged in (`huggingface-cli long`) and want to use private
    or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models).

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    module_file_or_url = os.path.join(pretrained_model_name_or_path,
        module_file)
    if os.path.isfile(module_file_or_url):
        resolved_module_file = module_file_or_url
        submodule = 'local'
    elif pretrained_model_name_or_path.count('/') == 0:
        available_versions = get_diffusers_versions()
        latest_version = 'v' + '.'.join(__version__.split('.')[:3])
        if revision is None:
            revision = (latest_version if latest_version in
                available_versions else 'main')
            logger.info(f'Defaulting to latest_version: {revision}.')
        elif revision in available_versions:
            revision = f'v{revision}'
        elif revision == 'main':
            revision = revision
        else:
            raise ValueError(
                f"`custom_revision`: {revision} does not exist. Please make sure to choose one of {', '.join(available_versions + ['main'])}."
                )
        github_url = COMMUNITY_PIPELINES_URL.format(revision=revision,
            pipeline=pretrained_model_name_or_path)
        try:
            resolved_module_file = cached_download(github_url, cache_dir=
                cache_dir, force_download=force_download, proxies=proxies,
                resume_download=resume_download, local_files_only=
                local_files_only, use_auth_token=False)
            submodule = 'git'
            module_file = pretrained_model_name_or_path + '.py'
        except EnvironmentError:
            logger.error(
                f'Could not locate the {module_file} inside {pretrained_model_name_or_path}.'
                )
            raise
    else:
        try:
            resolved_module_file = hf_hub_download(
                pretrained_model_name_or_path, module_file, cache_dir=
                cache_dir, force_download=force_download, proxies=proxies,
                resume_download=resume_download, local_files_only=
                local_files_only, use_auth_token=use_auth_token)
            submodule = os.path.join('local', '--'.join(
                pretrained_model_name_or_path.split('/')))
        except EnvironmentError:
            logger.error(
                f'Could not locate the {module_file} inside {pretrained_model_name_or_path}.'
                )
            raise
    modules_needed = check_imports(resolved_module_file)
    full_submodule = DIFFUSERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(HF_MODULES_CACHE) / full_submodule
    if submodule == 'local' or submodule == 'git':
        shutil.copy(resolved_module_file, submodule_path / module_file)
        for module_needed in modules_needed:
            module_needed = f'{module_needed}.py'
            shutil.copy(os.path.join(pretrained_model_name_or_path,
                module_needed), submodule_path / module_needed)
    else:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        elif use_auth_token is True:
            token = HfFolder.get_token()
        else:
            token = None
        commit_hash = model_info(pretrained_model_name_or_path, revision=
            revision, token=token).sha
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        create_dynamic_module(full_submodule)
        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
        for module_needed in modules_needed:
            if not (submodule_path / module_needed).exists():
                get_cached_module_file(pretrained_model_name_or_path,
                    f'{module_needed}.py', cache_dir=cache_dir,
                    force_download=force_download, resume_download=
                    resume_download, proxies=proxies, use_auth_token=
                    use_auth_token, revision=revision, local_files_only=
                    local_files_only)
    return os.path.join(full_submodule, module_file)
