def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[
    str, Dict[str, torch.Tensor]], adapter_name: Optional[str]=None, **kwargs):
    """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        pretrained_model_name_or_path_or_dict = (
            pretrained_model_name_or_path_or_dict.copy())
    state_dict, network_alphas = self.lora_state_dict(
        pretrained_model_name_or_path_or_dict, unet_config=self.unet.config,
        **kwargs)
    is_correct_format = all('lora' in key or 'dora_scale' in key for key in
        state_dict.keys())
    if not is_correct_format:
        raise ValueError('Invalid LoRA checkpoint.')
    self.load_lora_into_unet(state_dict, network_alphas=network_alphas,
        unet=self.unet, adapter_name=adapter_name, _pipeline=self)
    text_encoder_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder.' in k}
    if len(text_encoder_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder,
            prefix='text_encoder', lora_scale=self.lora_scale, adapter_name
            =adapter_name, _pipeline=self)
    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder_2.' in k}
    if len(text_encoder_2_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_2_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder_2,
            prefix='text_encoder_2', lora_scale=self.lora_scale,
            adapter_name=adapter_name, _pipeline=self)
