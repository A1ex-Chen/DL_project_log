def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[
    str, Dict[str, torch.Tensor]], **kwargs):
    if is_accelerate_available() and is_accelerate_version('>=', '0.17.0.dev0'
        ):
        from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
    else:
        raise ImportError('Offloading requires `accelerate v0.17.0` or higher.'
            )
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    recursive = False
    for _, component in self.components.items():
        if isinstance(component, torch.nn.Module):
            if hasattr(component, '_hf_hook'):
                is_model_cpu_offload = isinstance(getattr(component,
                    '_hf_hook'), CpuOffload)
                is_sequential_cpu_offload = isinstance(getattr(component,
                    '_hf_hook'), AlignDevicesHook) or hasattr(component.
                    _hf_hook, 'hooks') and isinstance(component._hf_hook.
                    hooks[0], AlignDevicesHook)
                logger.info(
                    'Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again.'
                    )
                recursive = is_sequential_cpu_offload
                remove_hook_from_module(component, recurse=recursive)
    state_dict, network_alphas = self.lora_state_dict(
        pretrained_model_name_or_path_or_dict, unet_config=self.unet.config,
        **kwargs)
    self.load_lora_into_unet(state_dict, network_alphas=network_alphas,
        unet=self.unet)
    text_encoder_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder.' in k}
    if len(text_encoder_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder,
            prefix='text_encoder', lora_scale=self.lora_scale)
    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder_2.' in k}
    if len(text_encoder_2_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_2_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder_2,
            prefix='text_encoder_2', lora_scale=self.lora_scale)
    if is_model_cpu_offload:
        self.enable_model_cpu_offload()
    elif is_sequential_cpu_offload:
        self.enable_sequential_cpu_offload()
