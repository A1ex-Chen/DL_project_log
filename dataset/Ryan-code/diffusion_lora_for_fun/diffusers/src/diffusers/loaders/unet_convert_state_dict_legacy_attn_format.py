def convert_state_dict_legacy_attn_format(self, state_dict, network_alphas):
    is_new_lora_format = all(key.startswith(self.unet_name) or key.
        startswith(self.text_encoder_name) for key in state_dict.keys())
    if is_new_lora_format:
        is_text_encoder_present = any(key.startswith(self.text_encoder_name
            ) for key in state_dict.keys())
        if is_text_encoder_present:
            warn_message = (
                'The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights).'
                )
            logger.warning(warn_message)
        unet_keys = [k for k in state_dict.keys() if k.startswith(self.
            unet_name)]
        state_dict = {k.replace(f'{self.unet_name}.', ''): v for k, v in
            state_dict.items() if k in unet_keys}
    if any('processor' in k.split('.') for k in state_dict.keys()):

        def format_to_lora_compatible(key):
            if 'processor' not in key.split('.'):
                return key
            return key.replace('.processor', '').replace('to_out_lora',
                'to_out.0.lora').replace('_lora', '.lora')
        state_dict = {format_to_lora_compatible(k): v for k, v in
            state_dict.items()}
        if network_alphas is not None:
            network_alphas = {format_to_lora_compatible(k): v for k, v in
                network_alphas.items()}
    return state_dict, network_alphas
