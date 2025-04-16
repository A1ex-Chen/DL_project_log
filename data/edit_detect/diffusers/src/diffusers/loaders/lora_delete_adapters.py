def delete_adapters(self, adapter_names: Union[List[str], str]):
    """
        Args:
        Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).
            adapter_names (`Union[List[str], str]`):
                The names of the adapter to delete. Can be a single string or a list of strings
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    unet.delete_adapters(adapter_names)
    for adapter_name in adapter_names:
        if hasattr(self, 'text_encoder'):
            delete_adapter_layers(self.text_encoder, adapter_name)
        if hasattr(self, 'text_encoder_2'):
            delete_adapter_layers(self.text_encoder_2, adapter_name)
