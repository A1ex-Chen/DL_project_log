def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[
    str, Dict[str, torch.Tensor]], **kwargs):
    """
        Load pretrained LoRA attention processor layers into [`UNet2DConditionModel`] and
        [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).

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
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
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
    self._lora_scale = 1.0
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
    network_alpha = None
    if all(k.startswith('lora_te_') or k.startswith('lora_unet_') for k in
        state_dict.keys()):
        state_dict, network_alpha = self._convert_kohya_lora_to_diffusers(
            state_dict)
    keys = list(state_dict.keys())
    if all(key.startswith(self.unet_name) or key.startswith(self.
        text_encoder_name) for key in keys):
        unet_keys = [k for k in keys if k.startswith(self.unet_name)]
        logger.info(f'Loading {self.unet_name}.')
        unet_lora_state_dict = {k.replace(f'{self.unet_name}.', ''): v for 
            k, v in state_dict.items() if k in unet_keys}
        self.unet.load_attn_procs(unet_lora_state_dict, network_alpha=
            network_alpha)
        text_encoder_keys = [k for k in keys if k.startswith(self.
            text_encoder_name)]
        text_encoder_lora_state_dict = {k.replace(
            f'{self.text_encoder_name}.', ''): v for k, v in state_dict.
            items() if k in text_encoder_keys}
        if len(text_encoder_lora_state_dict) > 0:
            logger.info(f'Loading {self.text_encoder_name}.')
            attn_procs_text_encoder = self._load_text_encoder_attn_procs(
                text_encoder_lora_state_dict, network_alpha=network_alpha)
            self._modify_text_encoder(attn_procs_text_encoder)
            self._text_encoder_lora_attn_procs = attn_procs_text_encoder
    elif not all(key.startswith(self.unet_name) or key.startswith(self.
        text_encoder_name) for key in state_dict.keys()):
        self.unet.load_attn_procs(state_dict)
        warn_message = (
            "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."
            )
        warnings.warn(warn_message)
