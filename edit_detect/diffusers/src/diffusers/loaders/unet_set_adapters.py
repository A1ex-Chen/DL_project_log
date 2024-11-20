def set_adapters(self, adapter_names: Union[List[str], str], weights:
    Optional[Union[float, Dict, List[float], List[Dict], List[None]]]=None):
    """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for `set_adapters()`.')
    adapter_names = [adapter_names] if isinstance(adapter_names, str
        ) else adapter_names
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError(
            f'Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}.'
            )
    weights = [(w if w is not None else 1.0) for w in weights]
    weights = _maybe_expand_lora_scales(self, weights)
    set_weights_and_activate_adapters(self, adapter_names, weights)
