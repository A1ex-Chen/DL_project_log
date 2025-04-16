def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str,
    Dict[str, torch.Tensor]], **kwargs):
    """
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`cross_attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

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
    is_custom_diffusion = any('custom_diffusion' in k for k in state_dict.
        keys())
    if is_lora:
        is_new_lora_format = all(key.startswith(self.unet_name) or key.
            startswith(self.text_encoder_name) for key in state_dict.keys())
        if is_new_lora_format:
            is_text_encoder_present = any(key.startswith(self.
                text_encoder_name) for key in state_dict.keys())
            if is_text_encoder_present:
                warn_message = (
                    'The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights).'
                    )
                warnings.warn(warn_message)
            unet_keys = [k for k in state_dict.keys() if k.startswith(self.
                unet_name)]
            state_dict = {k.replace(f'{self.unet_name}.', ''): v for k, v in
                state_dict.items() if k in unet_keys}
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = '.'.join(key.split('.')[:-3]
                ), '.'.join(key.split('.')[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value
        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict['to_k_lora.down.weight'].shape[0]
            hidden_size = value_dict['to_k_lora.up.weight'].shape[0]
            attn_processor = self
            for sub_key in key.split('.'):
                attn_processor = getattr(attn_processor, sub_key)
            if isinstance(attn_processor, (AttnAddedKVProcessor,
                SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                cross_attention_dim = value_dict['add_k_proj_lora.down.weight'
                    ].shape[1]
                attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                cross_attention_dim = value_dict['to_k_lora.down.weight'
                    ].shape[1]
                if isinstance(attn_processor, (XFormersAttnProcessor,
                    LoRAXFormersAttnProcessor)):
                    attn_processor_class = LoRAXFormersAttnProcessor
                else:
                    attn_processor_class = LoRAAttnProcessor2_0 if hasattr(F,
                        'scaled_dot_product_attention') else LoRAAttnProcessor
            attn_processors[key] = attn_processor_class(hidden_size=
                hidden_size, cross_attention_dim=cross_attention_dim, rank=
                rank, network_alpha=network_alpha)
            attn_processors[key].load_state_dict(value_dict)
    elif is_custom_diffusion:
        custom_diffusion_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            if len(value) == 0:
                custom_diffusion_grouped_dict[key] = {}
            else:
                if 'to_out' in key:
                    attn_processor_key, sub_key = '.'.join(key.split('.')[:-3]
                        ), '.'.join(key.split('.')[-3:])
                else:
                    attn_processor_key, sub_key = '.'.join(key.split('.')[:-2]
                        ), '.'.join(key.split('.')[-2:])
                custom_diffusion_grouped_dict[attn_processor_key][sub_key
                    ] = value
        for key, value_dict in custom_diffusion_grouped_dict.items():
            if len(value_dict) == 0:
                attn_processors[key] = CustomDiffusionAttnProcessor(train_kv
                    =False, train_q_out=False, hidden_size=None,
                    cross_attention_dim=None)
            else:
                cross_attention_dim = value_dict['to_k_custom_diffusion.weight'
                    ].shape[1]
                hidden_size = value_dict['to_k_custom_diffusion.weight'].shape[
                    0]
                train_q_out = (True if 'to_q_custom_diffusion.weight' in
                    value_dict else False)
                attn_processors[key] = CustomDiffusionAttnProcessor(train_kv
                    =True, train_q_out=train_q_out, hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim)
                attn_processors[key].load_state_dict(value_dict)
    else:
        raise ValueError(
            f'{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training.'
            )
    attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k,
        v in attn_processors.items()}
    self.set_attn_processor(attn_processors)
