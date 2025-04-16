@classmethod
@validate_hf_hub_args
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
    cache_dir = kwargs.pop('cache_dir', None)
    ignore_mismatched_sizes = kwargs.pop('ignore_mismatched_sizes', False)
    force_download = kwargs.pop('force_download', False)
    from_flax = kwargs.pop('from_flax', False)
    resume_download = kwargs.pop('resume_download', None)
    proxies = kwargs.pop('proxies', None)
    output_loading_info = kwargs.pop('output_loading_info', False)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    torch_dtype = kwargs.pop('torch_dtype', None)
    subfolder = kwargs.pop('subfolder', None)
    device_map = kwargs.pop('device_map', None)
    max_memory = kwargs.pop('max_memory', None)
    offload_folder = kwargs.pop('offload_folder', None)
    offload_state_dict = kwargs.pop('offload_state_dict', False)
    low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage',
        _LOW_CPU_MEM_USAGE_DEFAULT)
    variant = kwargs.pop('variant', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True
    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            """Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: 
```
pip install accelerate
```
."""
            )
    if device_map is not None and not is_accelerate_available():
        raise NotImplementedError(
            'Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set `device_map=None`. You can install accelerate with `pip install accelerate`.'
            )
    if device_map is not None and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set `device_map=None`.'
            )
    if low_cpu_mem_usage is True and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`.'
            )
    if low_cpu_mem_usage is False and device_map is not None:
        raise ValueError(
            f'You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and dispatching. Please make sure to set `low_cpu_mem_usage=True`.'
            )
    if isinstance(device_map, torch.device):
        device_map = {'': device_map}
    elif isinstance(device_map, str) and device_map not in ['auto',
        'balanced', 'balanced_low_0', 'sequential']:
        try:
            device_map = {'': torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                f"When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or 'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
        else:
            device_map = {'': device_map}
    if device_map is not None:
        if low_cpu_mem_usage is None:
            low_cpu_mem_usage = True
        elif not low_cpu_mem_usage:
            raise ValueError(
                'Passing along a `device_map` requires `low_cpu_mem_usage=True`'
                )
    if low_cpu_mem_usage:
        if device_map is not None and not is_torch_version('>=', '1.10'):
            raise ValueError(
                '`low_cpu_mem_usage` and `device_map` require PyTorch >= 1.10.'
                )
    config_path = pretrained_model_name_or_path
    user_agent = {'diffusers': __version__, 'file_type': 'model',
        'framework': 'pytorch'}
    config, unused_kwargs, commit_hash = cls.load_config(config_path,
        cache_dir=cache_dir, return_unused_kwargs=True, return_commit_hash=
        True, force_download=force_download, resume_download=
        resume_download, proxies=proxies, local_files_only=local_files_only,
        token=token, revision=revision, subfolder=subfolder, user_agent=
        user_agent, **kwargs)
    model_file = None
    if from_flax:
        model_file = _get_model_file(pretrained_model_name_or_path,
            weights_name=FLAX_WEIGHTS_NAME, cache_dir=cache_dir,
            force_download=force_download, resume_download=resume_download,
            proxies=proxies, local_files_only=local_files_only, token=token,
            revision=revision, subfolder=subfolder, user_agent=user_agent,
            commit_hash=commit_hash)
        model = cls.from_config(config, **unused_kwargs)
        from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model
        model = load_flax_checkpoint_in_pytorch_model(model, model_file)
    else:
        if use_safetensors:
            try:
                model_file = _get_model_file(pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME,
                    variant), cache_dir=cache_dir, force_download=
                    force_download, resume_download=resume_download,
                    proxies=proxies, local_files_only=local_files_only,
                    token=token, revision=revision, subfolder=subfolder,
                    user_agent=user_agent, commit_hash=commit_hash)
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
        if model_file is None:
            model_file = _get_model_file(pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant), cache_dir
                =cache_dir, force_download=force_download, resume_download=
                resume_download, proxies=proxies, local_files_only=
                local_files_only, token=token, revision=revision, subfolder
                =subfolder, user_agent=user_agent, commit_hash=commit_hash)
        if low_cpu_mem_usage:
            with accelerate.init_empty_weights():
                model = cls.from_config(config, **unused_kwargs)
            if device_map is None:
                param_device = 'cpu'
                state_dict = load_state_dict(model_file, variant=variant)
                model._convert_deprecated_attention_blocks(state_dict)
                missing_keys = set(model.state_dict().keys()) - set(state_dict
                    .keys())
                if len(missing_keys) > 0:
                    raise ValueError(
                        f"""Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are missing: 
 {', '.join(missing_keys)}. 
 Please make sure to pass `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize those weights or else make sure your checkpoint file is correct."""
                        )
                unexpected_keys = load_model_dict_into_meta(model,
                    state_dict, device=param_device, dtype=torch_dtype,
                    model_name_or_path=pretrained_model_name_or_path)
                if cls._keys_to_ignore_on_load_unexpected is not None:
                    for pat in cls._keys_to_ignore_on_load_unexpected:
                        unexpected_keys = [k for k in unexpected_keys if re
                            .search(pat, k) is None]
                if len(unexpected_keys) > 0:
                    logger.warning(
                        f"""Some weights of the model checkpoint were not used when initializing {cls.__name__}: 
 {[', '.join(unexpected_keys)]}"""
                        )
            else:
                device_map = _determine_device_map(model, device_map,
                    max_memory, torch_dtype)
                try:
                    accelerate.load_checkpoint_and_dispatch(model,
                        model_file, device_map, max_memory=max_memory,
                        offload_folder=offload_folder, offload_state_dict=
                        offload_state_dict, dtype=torch_dtype, force_hooks=
                        True, strict=True)
                except AttributeError as e:
                    if "'Attention' object has no attribute" in str(e):
                        logger.warning(
                            f"Taking `{str(e)}` while using `accelerate.load_checkpoint_and_dispatch` to mean {pretrained_model_name_or_path} was saved with deprecated attention block weight names. We will load it with the deprecated attention block names and convert them on the fly to the new attention block format. Please re-save the model after this conversion, so we don't have to do the on the fly renaming in the future. If the model is from a hub checkpoint, please also re-upload it or open a PR on the original repository."
                            )
                        model._temp_convert_self_to_deprecated_attention_blocks(
                            )
                        accelerate.load_checkpoint_and_dispatch(model,
                            model_file, device_map, max_memory=max_memory,
                            offload_folder=offload_folder,
                            offload_state_dict=offload_state_dict, dtype=
                            torch_dtype)
                        model._undo_temp_convert_self_to_deprecated_attention_blocks(
                            )
                    else:
                        raise e
            loading_info = {'missing_keys': [], 'unexpected_keys': [],
                'mismatched_keys': [], 'error_msgs': []}
        else:
            model = cls.from_config(config, **unused_kwargs)
            state_dict = load_state_dict(model_file, variant=variant)
            model._convert_deprecated_attention_blocks(state_dict)
            (model, missing_keys, unexpected_keys, mismatched_keys, error_msgs
                ) = (cls._load_pretrained_model(model, state_dict,
                model_file, pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes))
            loading_info = {'missing_keys': missing_keys, 'unexpected_keys':
                unexpected_keys, 'mismatched_keys': mismatched_keys,
                'error_msgs': error_msgs}
    if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
        raise ValueError(
            f'{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}.'
            )
    elif torch_dtype is not None:
        model = model.to(torch_dtype)
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    model.eval()
    if output_loading_info:
        return model, loading_info
    return model
