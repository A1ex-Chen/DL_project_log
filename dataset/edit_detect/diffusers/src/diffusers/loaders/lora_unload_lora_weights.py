def unload_lora_weights(self):
    """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    if not USE_PEFT_BACKEND:
        if version.parse(__version__) > version.parse('0.23'):
            logger.warning(
                'You are using `unload_lora_weights` to disable and unload lora weights. If you want to iteratively enable and disable adapter weights,you can use `pipe.enable_lora()` or `pipe.disable_lora()`. After installing the latest version of PEFT.'
                )
        for _, module in unet.named_modules():
            if hasattr(module, 'set_lora_layer'):
                module.set_lora_layer(None)
    else:
        recurse_remove_peft_layers(unet)
        if hasattr(unet, 'peft_config'):
            del unet.peft_config
    self._remove_text_encoder_monkey_patch()
