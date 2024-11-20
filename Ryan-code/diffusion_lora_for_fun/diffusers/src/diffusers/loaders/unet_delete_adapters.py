def delete_adapters(self, adapter_names: Union[List[str], str]):
    """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    for adapter_name in adapter_names:
        delete_adapter_layers(self, adapter_name)
        if hasattr(self, 'peft_config'):
            self.peft_config.pop(adapter_name, None)
