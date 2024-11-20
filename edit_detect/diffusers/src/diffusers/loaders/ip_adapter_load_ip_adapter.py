@validate_hf_hub_args
def load_ip_adapter(self, pretrained_model_name_or_path_or_dict: Union[str,
    List[str], Dict[str, torch.Tensor]], subfolder: Union[str, List[str]],
    weight_name: Union[str, List[str]], image_encoder_folder: Optional[str]
    ='image_encoder', **kwargs):
    """
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            subfolder (`str` or `List[str]`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a
                list is passed, it should have the same length as `weight_name`.
            weight_name (`str` or `List[str]`):
                The name of the weight file to load. If a list is passed, it should have the same length as
                `weight_name`.
            image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
                The subfolder location of the image encoder within a larger model repository on the Hub or locally.
                Pass `None` to not load the image encoder. If the image encoder is located in a folder inside
                `subfolder`, you only need to pass the name of the folder that contains image encoder weights, e.g.
                `image_encoder_folder="image_encoder"`. If the image encoder is located in a folder other than
                `subfolder`, you should pass the path to the folder that contains image encoder weights, for example,
                `image_encoder_folder="different_subfolder/image_encoder"`.
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
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
        """
    if not isinstance(weight_name, list):
        weight_name = [weight_name]
    if not isinstance(pretrained_model_name_or_path_or_dict, list):
        pretrained_model_name_or_path_or_dict = [
            pretrained_model_name_or_path_or_dict]
    if len(pretrained_model_name_or_path_or_dict) == 1:
        pretrained_model_name_or_path_or_dict = (
            pretrained_model_name_or_path_or_dict * len(weight_name))
    if not isinstance(subfolder, list):
        subfolder = [subfolder]
    if len(subfolder) == 1:
        subfolder = subfolder * len(weight_name)
    if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
        raise ValueError(
            '`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.'
            )
    if len(weight_name) != len(subfolder):
        raise ValueError(
            '`weight_name` and `subfolder` must have the same length.')
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage',
        _LOW_CPU_MEM_USAGE_DEFAULT)
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
    user_agent = {'file_type': 'attn_procs_weights', 'framework': 'pytorch'}
    state_dicts = []
    for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder):
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(pretrained_model_name_or_path_or_dict,
                weights_name=weight_name, cache_dir=cache_dir,
                force_download=force_download, resume_download=
                resume_download, proxies=proxies, local_files_only=
                local_files_only, token=token, revision=revision, subfolder
                =subfolder, user_agent=user_agent)
            if weight_name.endswith('.safetensors'):
                state_dict = {'image_proj': {}, 'ip_adapter': {}}
                with safe_open(model_file, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        if key.startswith('image_proj.'):
                            state_dict['image_proj'][key.replace(
                                'image_proj.', '')] = f.get_tensor(key)
                        elif key.startswith('ip_adapter.'):
                            state_dict['ip_adapter'][key.replace(
                                'ip_adapter.', '')] = f.get_tensor(key)
            else:
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict
        keys = list(state_dict.keys())
        if keys != ['image_proj', 'ip_adapter']:
            raise ValueError(
                'Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.'
                )
        state_dicts.append(state_dict)
        if hasattr(self, 'image_encoder') and getattr(self, 'image_encoder',
            None) is None:
            if image_encoder_folder is not None:
                if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                    logger.info(
                        f'loading image_encoder from {pretrained_model_name_or_path_or_dict}'
                        )
                    if image_encoder_folder.count('/') == 0:
                        image_encoder_subfolder = Path(subfolder,
                            image_encoder_folder).as_posix()
                    else:
                        image_encoder_subfolder = Path(image_encoder_folder
                            ).as_posix()
                    image_encoder = (CLIPVisionModelWithProjection.
                        from_pretrained(
                        pretrained_model_name_or_path_or_dict, subfolder=
                        image_encoder_subfolder, low_cpu_mem_usage=
                        low_cpu_mem_usage).to(self.device, dtype=self.dtype))
                    self.register_modules(image_encoder=image_encoder)
                else:
                    raise ValueError(
                        '`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict.'
                        )
            else:
                logger.warning(
                    'image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter.Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead.'
                    )
        if hasattr(self, 'feature_extractor') and getattr(self,
            'feature_extractor', None) is None:
            feature_extractor = CLIPImageProcessor()
            self.register_modules(feature_extractor=feature_extractor)
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=
        low_cpu_mem_usage)
    extra_loras = unet._load_ip_adapter_loras(state_dicts)
    if extra_loras != {}:
        if not USE_PEFT_BACKEND:
            logger.warning('PEFT backend is required to load these weights.')
        else:
            peft_config = getattr(unet, 'peft_config', {})
            for k, lora in extra_loras.items():
                if f'faceid_{k}' not in peft_config:
                    self.load_lora_weights(lora, adapter_name=f'faceid_{k}')
                    self.set_adapters([f'faceid_{k}'], adapter_weights=[1.0])
