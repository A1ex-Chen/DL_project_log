@classmethod
@validate_hf_hub_args
def lora_state_dict(cls, pretrained_model_name_or_path_or_dict: Union[str,
    Dict[str, torch.Tensor]], **kwargs):
    """
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
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
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    weight_name = kwargs.pop('weight_name', None)
    unet_config = kwargs.pop('unet_config', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True
    user_agent = {'file_type': 'attn_procs_weights', 'framework': 'pytorch'}
    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        if (use_safetensors and weight_name is None or weight_name is not
            None and weight_name.endswith('.safetensors')):
            try:
                if weight_name is None:
                    weight_name = cls._best_guess_weight_name(
                        pretrained_model_name_or_path_or_dict,
                        file_extension='.safetensors', local_files_only=
                        local_files_only)
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict, weights_name=
                    weight_name or LORA_WEIGHT_NAME_SAFE, cache_dir=
                    cache_dir, force_download=force_download,
                    resume_download=resume_download, proxies=proxies,
                    local_files_only=local_files_only, token=token,
                    revision=revision, subfolder=subfolder, user_agent=
                    user_agent)
                state_dict = safetensors.torch.load_file(model_file, device
                    ='cpu')
            except (IOError, safetensors.SafetensorError) as e:
                if not allow_pickle:
                    raise e
                model_file = None
                pass
        if model_file is None:
            if weight_name is None:
                weight_name = cls._best_guess_weight_name(
                    pretrained_model_name_or_path_or_dict, file_extension=
                    '.bin', local_files_only=local_files_only)
            model_file = _get_model_file(pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME, cache_dir=
                cache_dir, force_download=force_download, resume_download=
                resume_download, proxies=proxies, local_files_only=
                local_files_only, token=token, revision=revision, subfolder
                =subfolder, user_agent=user_agent)
            state_dict = load_state_dict(model_file)
    else:
        state_dict = pretrained_model_name_or_path_or_dict
    network_alphas = None
    if all(k.startswith('lora_te_') or k.startswith('lora_unet_') or k.
        startswith('lora_te1_') or k.startswith('lora_te2_') for k in
        state_dict.keys()):
        if unet_config is not None:
            state_dict = _maybe_map_sgm_blocks_to_diffusers(state_dict,
                unet_config)
        state_dict, network_alphas = _convert_kohya_lora_to_diffusers(
            state_dict)
    return state_dict, network_alphas
