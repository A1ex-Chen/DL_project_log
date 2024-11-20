@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str,
    os.PathLike]], **kwargs):
    """
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
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
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_state_dict (`bool`, *optional*):
                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
                `True` when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights will be downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model will be forcibly loaded from
                `safetensors` weights. If set to `False`, loading will *not* use `safetensors`.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        """
    cache_dir = kwargs.pop('cache_dir', DIFFUSERS_CACHE)
    ignore_mismatched_sizes = kwargs.pop('ignore_mismatched_sizes', False)
    force_download = kwargs.pop('force_download', False)
    from_flax = kwargs.pop('from_flax', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    output_loading_info = kwargs.pop('output_loading_info', False)
    local_files_only = kwargs.pop('local_files_only', HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop('use_auth_token', None)
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
    if use_safetensors and not is_safetensors_available():
        raise ValueError(
            '`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetenstors'
            )
    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = is_safetensors_available()
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
    config_path = pretrained_model_name_or_path
    user_agent = {'diffusers': __version__, 'file_type': 'model',
        'framework': 'pytorch'}
    config, unused_kwargs, commit_hash = cls.load_config(config_path,
        cache_dir=cache_dir, return_unused_kwargs=True, return_commit_hash=
        True, force_download=force_download, resume_download=
        resume_download, proxies=proxies, local_files_only=local_files_only,
        use_auth_token=use_auth_token, revision=revision, subfolder=
        subfolder, device_map=device_map, max_memory=max_memory,
        offload_folder=offload_folder, offload_state_dict=
        offload_state_dict, user_agent=user_agent, **kwargs)
    model_file = None
    if from_flax:
        model_file = _get_model_file(pretrained_model_name_or_path,
            weights_name=FLAX_WEIGHTS_NAME, cache_dir=cache_dir,
            force_download=force_download, resume_download=resume_download,
            proxies=proxies, local_files_only=local_files_only,
            use_auth_token=use_auth_token, revision=revision, subfolder=
            subfolder, user_agent=user_agent, commit_hash=commit_hash)
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
                    use_auth_token=use_auth_token, revision=revision,
                    subfolder=subfolder, user_agent=user_agent, commit_hash
                    =commit_hash)
            except IOError as e:
                if not allow_pickle:
                    raise e
                pass
        if model_file is None:
            model_file = _get_model_file(pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant), cache_dir
                =cache_dir, force_download=force_download, resume_download=
                resume_download, proxies=proxies, local_files_only=
                local_files_only, use_auth_token=use_auth_token, revision=
                revision, subfolder=subfolder, user_agent=user_agent,
                commit_hash=commit_hash)
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
                unexpected_keys = []
                empty_state_dict = model.state_dict()
                for param_name, param in state_dict.items():
                    accepts_dtype = 'dtype' in set(inspect.signature(
                        set_module_tensor_to_device).parameters.keys())
                    if param_name not in empty_state_dict:
                        unexpected_keys.append(param_name)
                        continue
                    if empty_state_dict[param_name].shape != param.shape:
                        raise ValueError(
                            f'Cannot load {pretrained_model_name_or_path} because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example.'
                            )
                    if accepts_dtype:
                        set_module_tensor_to_device(model, param_name,
                            param_device, value=param, dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, param_name,
                            param_device, value=param)
                if cls._keys_to_ignore_on_load_unexpected is not None:
                    for pat in cls._keys_to_ignore_on_load_unexpected:
                        unexpected_keys = [k for k in unexpected_keys if re
                            .search(pat, k) is None]
                if len(unexpected_keys) > 0:
                    logger.warn(
                        f"""Some weights of the model checkpoint were not used when initializing {cls.__name__}: 
 {[', '.join(unexpected_keys)]}"""
                        )
            else:
                try:
                    accelerate.load_checkpoint_and_dispatch(model,
                        model_file, device_map, max_memory=max_memory,
                        offload_folder=offload_folder, offload_state_dict=
                        offload_state_dict, dtype=torch_dtype)
                except AttributeError as e:
                    if "'Attention' object has no attribute" in str(e):
                        logger.warn(
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
