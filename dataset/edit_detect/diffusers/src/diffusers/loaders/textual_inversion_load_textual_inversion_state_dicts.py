@validate_hf_hub_args
def load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs
    ):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    weight_name = kwargs.pop('weight_name', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True
    user_agent = {'file_type': 'text_inversion', 'framework': 'pytorch'}
    state_dicts = []
    for pretrained_model_name_or_path in pretrained_model_name_or_paths:
        if not isinstance(pretrained_model_name_or_path, (dict, torch.Tensor)):
            model_file = None
            if (use_safetensors and weight_name is None or weight_name is not
                None and weight_name.endswith('.safetensors')):
                try:
                    model_file = _get_model_file(pretrained_model_name_or_path,
                        weights_name=weight_name or
                        TEXT_INVERSION_NAME_SAFE, cache_dir=cache_dir,
                        force_download=force_download, resume_download=
                        resume_download, proxies=proxies, local_files_only=
                        local_files_only, token=token, revision=revision,
                        subfolder=subfolder, user_agent=user_agent)
                    state_dict = safetensors.torch.load_file(model_file,
                        device='cpu')
                except Exception as e:
                    if not allow_pickle:
                        raise e
                    model_file = None
            if model_file is None:
                model_file = _get_model_file(pretrained_model_name_or_path,
                    weights_name=weight_name or TEXT_INVERSION_NAME,
                    cache_dir=cache_dir, force_download=force_download,
                    resume_download=resume_download, proxies=proxies,
                    local_files_only=local_files_only, token=token,
                    revision=revision, subfolder=subfolder, user_agent=
                    user_agent)
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path
        state_dicts.append(state_dict)
    return state_dicts
