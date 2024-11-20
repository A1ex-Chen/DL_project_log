@classmethod
def load_lora_into_text_encoder(cls, state_dict, network_alphas,
    text_encoder, prefix=None, lora_scale=1.0, low_cpu_mem_usage=None,
    adapter_name=None, _pipeline=None):
    """
        This will load the LoRA layers specified in `state_dict` into `text_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key should be prefixed with an
                additional `text_encoder` to distinguish between unet lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            text_encoder (`CLIPTextModel`):
                The text encoder model to load the LoRA layers into.
            prefix (`str`):
                Expected prefix of the `text_encoder` in the `state_dict`.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
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
    from peft import LoraConfig
    low_cpu_mem_usage = (low_cpu_mem_usage if low_cpu_mem_usage is not None
         else _LOW_CPU_MEM_USAGE_DEFAULT)
    keys = list(state_dict.keys())
    prefix = cls.text_encoder_name if prefix is None else prefix
    if any(cls.text_encoder_name in key for key in keys):
        text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.
            split('.')[0] == prefix]
        text_encoder_lora_state_dict = {k.replace(f'{prefix}.', ''): v for 
            k, v in state_dict.items() if k in text_encoder_keys}
        if len(text_encoder_lora_state_dict) > 0:
            logger.info(f'Loading {prefix}.')
            rank = {}
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                text_encoder_lora_state_dict)
            text_encoder_lora_state_dict = convert_state_dict_to_peft(
                text_encoder_lora_state_dict)
            for name, _ in text_encoder_attn_modules(text_encoder):
                rank_key = f'{name}.out_proj.lora_B.weight'
                rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1
                    ]
            patch_mlp = any('.mlp.' in key for key in
                text_encoder_lora_state_dict.keys())
            if patch_mlp:
                for name, _ in text_encoder_mlp_modules(text_encoder):
                    rank_key_fc1 = f'{name}.fc1.lora_B.weight'
                    rank_key_fc2 = f'{name}.fc2.lora_B.weight'
                    rank[rank_key_fc1] = text_encoder_lora_state_dict[
                        rank_key_fc1].shape[1]
                    rank[rank_key_fc2] = text_encoder_lora_state_dict[
                        rank_key_fc2].shape[1]
            if network_alphas is not None:
                alpha_keys = [k for k in network_alphas.keys() if k.
                    startswith(prefix) and k.split('.')[0] == prefix]
                network_alphas = {k.replace(f'{prefix}.', ''): v for k, v in
                    network_alphas.items() if k in alpha_keys}
            lora_config_kwargs = get_peft_kwargs(rank, network_alphas,
                text_encoder_lora_state_dict, is_unet=False)
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
                adapter_name = get_adapter_name(text_encoder)
            is_model_cpu_offload, is_sequential_cpu_offload = (cls.
                _optionally_disable_offloading(_pipeline))
            text_encoder.load_adapter(adapter_name=adapter_name,
                adapter_state_dict=text_encoder_lora_state_dict,
                peft_config=lora_config)
            scale_lora_layers(text_encoder, weight=lora_scale)
            text_encoder.to(device=text_encoder.device, dtype=text_encoder.
                dtype)
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
