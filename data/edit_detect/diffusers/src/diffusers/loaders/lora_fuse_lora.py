def fuse_lora(self, fuse_unet: bool=True, fuse_text_encoder: bool=True,
    lora_scale: float=1.0, safe_fusing: bool=False, adapter_names: Optional
    [List[str]]=None):
    """
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            fuse_unet (`bool`, defaults to `True`): Whether to fuse the UNet LoRA parameters.
            fuse_text_encoder (`bool`, defaults to `True`):
                Whether to fuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
    from peft.tuners.tuners_utils import BaseTunerLayer
    if fuse_unet or fuse_text_encoder:
        self.num_fused_loras += 1
        if self.num_fused_loras > 1:
            logger.warning(
                'The current API is supported for operating with a single LoRA file. You are trying to load and fuse more than one LoRA which is not well-supported.'
                )
    if fuse_unet:
        unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
            ) else self.unet
        unet.fuse_lora(lora_scale, safe_fusing=safe_fusing, adapter_names=
            adapter_names)

    def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=
        False, adapter_names=None):
        merge_kwargs = {'safe_merge': safe_fusing}
        for module in text_encoder.modules():
            if isinstance(module, BaseTunerLayer):
                if lora_scale != 1.0:
                    module.scale_layer(lora_scale)
                supported_merge_kwargs = list(inspect.signature(module.
                    merge).parameters)
                if 'adapter_names' in supported_merge_kwargs:
                    merge_kwargs['adapter_names'] = adapter_names
                elif 'adapter_names' not in supported_merge_kwargs and adapter_names is not None:
                    raise ValueError(
                        'The `adapter_names` argument is not supported with your PEFT version. Please upgrade to the latest version of PEFT. `pip install -U peft`'
                        )
                module.merge(**merge_kwargs)
    if fuse_text_encoder:
        if hasattr(self, 'text_encoder'):
            fuse_text_encoder_lora(self.text_encoder, lora_scale,
                safe_fusing, adapter_names=adapter_names)
        if hasattr(self, 'text_encoder_2'):
            fuse_text_encoder_lora(self.text_encoder_2, lora_scale,
                safe_fusing, adapter_names=adapter_names)
