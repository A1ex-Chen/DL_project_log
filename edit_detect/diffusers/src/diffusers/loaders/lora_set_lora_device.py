def set_lora_device(self, adapter_names: List[str], device: Union[torch.
    device, str, int]) ->None:
    """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[torch.device, str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    from peft.tuners.tuners_utils import BaseTunerLayer
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    for unet_module in unet.modules():
        if isinstance(unet_module, BaseTunerLayer):
            for adapter_name in adapter_names:
                unet_module.lora_A[adapter_name].to(device)
                unet_module.lora_B[adapter_name].to(device)
                if hasattr(unet_module, 'lora_magnitude_vector'
                    ) and unet_module.lora_magnitude_vector is not None:
                    unet_module.lora_magnitude_vector[adapter_name
                        ] = unet_module.lora_magnitude_vector[adapter_name].to(
                        device)
    modules_to_process = []
    if hasattr(self, 'text_encoder'):
        modules_to_process.append(self.text_encoder)
    if hasattr(self, 'text_encoder_2'):
        modules_to_process.append(self.text_encoder_2)
    for text_encoder in modules_to_process:
        for text_encoder_module in text_encoder.modules():
            if isinstance(text_encoder_module, BaseTunerLayer):
                for adapter_name in adapter_names:
                    text_encoder_module.lora_A[adapter_name].to(device)
                    text_encoder_module.lora_B[adapter_name].to(device)
                    if (hasattr(text_encoder_module,
                        'lora_magnitude_vector') and text_encoder_module.
                        lora_magnitude_vector is not None):
                        text_encoder_module.lora_magnitude_vector[adapter_name
                            ] = text_encoder_module.lora_magnitude_vector[
                            adapter_name].to(device)
