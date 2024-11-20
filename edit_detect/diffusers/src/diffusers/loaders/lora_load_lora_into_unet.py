@classmethod
def load_lora_into_unet(cls, state_dict, network_alphas, unet,
    low_cpu_mem_usage=None, adapter_name=None, _pipeline=None):
    """
        This will load the LoRA layers specified in `state_dict` into `unet`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    low_cpu_mem_usage = (low_cpu_mem_usage if low_cpu_mem_usage is not None
         else _LOW_CPU_MEM_USAGE_DEFAULT)
    keys = list(state_dict.keys())
    if all(key.startswith(cls.unet_name) or key.startswith(cls.
        text_encoder_name) for key in keys):
        logger.info(f'Loading {cls.unet_name}.')
        unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
        state_dict = {k.replace(f'{cls.unet_name}.', ''): v for k, v in
            state_dict.items() if k in unet_keys}
        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(
                cls.unet_name)]
            network_alphas = {k.replace(f'{cls.unet_name}.', ''): v for k,
                v in network_alphas.items() if k in alpha_keys}
    elif not USE_PEFT_BACKEND:
        warn_message = (
            "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
            )
        logger.warning(warn_message)
    if len(state_dict.keys()) > 0:
        if adapter_name in getattr(unet, 'peft_config', {}):
            raise ValueError(
                f'Adapter name {adapter_name} already in use in the Unet - please select a new adapter name.'
                )
        state_dict = convert_unet_state_dict_to_peft(state_dict)
        if network_alphas is not None:
            network_alphas = convert_unet_state_dict_to_peft(network_alphas)
        rank = {}
        for key, val in state_dict.items():
            if 'lora_B' in key:
                rank[key] = val.shape[1]
        lora_config_kwargs = get_peft_kwargs(rank, network_alphas,
            state_dict, is_unet=True)
        if 'use_dora' in lora_config_kwargs:
            if lora_config_kwargs['use_dora']:
                if is_peft_version('<', '0.9.0'):
                    raise ValueError(
                        'You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`.'
                        )
            elif is_peft_version('<', '0.9.0'):
                lora_config_kwargs.pop('use_dora')
        lora_config = LoraConfig(**lora_config_kwargs)
        if adapter_name is None:
            adapter_name = get_adapter_name(unet)
        is_model_cpu_offload, is_sequential_cpu_offload = (cls.
            _optionally_disable_offloading(_pipeline))
        inject_adapter_in_model(lora_config, unet, adapter_name=adapter_name)
        incompatible_keys = set_peft_model_state_dict(unet, state_dict,
            adapter_name)
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, 'unexpected_keys',
                None)
            if unexpected_keys:
                logger.warning(
                    f'Loading adapter weights from state_dict led to unexpected keys not found in the model:  {unexpected_keys}. '
                    )
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
    unet.load_attn_procs(state_dict, network_alphas=network_alphas,
        low_cpu_mem_usage=low_cpu_mem_usage, _pipeline=_pipeline)
