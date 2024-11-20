def get_list_adapters(self) ->Dict[str, List[str]]:
    """
        Gets the current list of all available adapters in the pipeline.
        """
    if not USE_PEFT_BACKEND:
        raise ValueError(
            'PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`'
            )
    set_adapters = {}
    if hasattr(self, 'text_encoder') and hasattr(self.text_encoder,
        'peft_config'):
        set_adapters['text_encoder'] = list(self.text_encoder.peft_config.
            keys())
    if hasattr(self, 'text_encoder_2') and hasattr(self.text_encoder_2,
        'peft_config'):
        set_adapters['text_encoder_2'] = list(self.text_encoder_2.
            peft_config.keys())
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    if hasattr(self, self.unet_name) and hasattr(unet, 'peft_config'):
        set_adapters[self.unet_name] = list(self.unet.peft_config.keys())
    return set_adapters
