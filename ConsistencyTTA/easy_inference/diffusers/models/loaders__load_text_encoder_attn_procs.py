def _load_text_encoder_attn_procs(self,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.
    Tensor]], **kwargs):
    """
        Load pretrained attention processor layers for
        [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).

        <Tip warning={true}>

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        Returns:
            `Dict[name, LoRAAttnProcessor]`: Mapping between the module names and their corresponding
            [`LoRAAttnProcessor`].

        <Tip>

        It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
        models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>
        """
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop('use_auth_token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    weight_name = kwargs.pop('weight_name', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    network_alpha = kwargs.pop('network_alpha', None)
    if use_safetensors and not is_safetensors_available():
        raise ValueError(
            '`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors'
            )
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = is_safetensors_available()
        allow_pickle = True
    user_agent = {'file_type': 'attn_procs_weights', 'framework': 'pytorch'}
    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        if (use_safetensors and weight_name is None or weight_name is not
            None and weight_name.endswith('.safetensors')):
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict, weights_name=
                    weight_name or LORA_WEIGHT_NAME_SAFE, cache_dir=
                    cache_dir, force_download=force_download,
                    resume_download=resume_download, proxies=proxies,
                    local_files_only=local_files_only, use_auth_token=
                    use_auth_token, revision=revision, subfolder=subfolder,
                    user_agent=user_agent)
                state_dict = safetensors.torch.load_file(model_file, device
                    ='cpu')
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
        if model_file is None:
            model_file = _get_model_file(pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME, cache_dir=
                cache_dir, force_download=force_download, resume_download=
                resume_download, proxies=proxies, local_files_only=
                local_files_only, use_auth_token=use_auth_token, revision=
                revision, subfolder=subfolder, user_agent=user_agent)
            state_dict = torch.load(model_file, map_location='cpu')
    else:
        state_dict = pretrained_model_name_or_path_or_dict
    attn_processors = {}
    is_lora = all('lora' in k for k in state_dict.keys())
    if is_lora:
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = '.'.join(key.split('.')[:-3]
                ), '.'.join(key.split('.')[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value
        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict['to_k_lora.down.weight'].shape[0]
            cross_attention_dim = value_dict['to_k_lora.down.weight'].shape[1]
            hidden_size = value_dict['to_k_lora.up.weight'].shape[0]
            attn_processor_class = LoRAAttnProcessor2_0 if hasattr(F,
                'scaled_dot_product_attention') else LoRAAttnProcessor
            attn_processors[key] = attn_processor_class(hidden_size=
                hidden_size, cross_attention_dim=cross_attention_dim, rank=
                rank, network_alpha=network_alpha)
            attn_processors[key].load_state_dict(value_dict)
    else:
        raise ValueError(
            f'{model_file} does not seem to be in the correct format expected by LoRA training.'
            )
    attn_processors = {k: v.to(device=self.device, dtype=self.text_encoder.
        dtype) for k, v in attn_processors.items()}
    return attn_processors
