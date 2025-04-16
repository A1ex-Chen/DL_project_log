def get_active_adapters(self) ->List[str]:
    """
        Gets the list of the current active adapters.

        Example:

        ```python
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
        ).to("cuda")
        pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipeline.get_active_adapters()
        ```
        """
    if not USE_PEFT_BACKEND:
        raise ValueError(
            'PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`'
            )
    from peft.tuners.tuners_utils import BaseTunerLayer
    active_adapters = []
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    for module in unet.modules():
        if isinstance(module, BaseTunerLayer):
            active_adapters = module.active_adapters
            break
    return active_adapters
