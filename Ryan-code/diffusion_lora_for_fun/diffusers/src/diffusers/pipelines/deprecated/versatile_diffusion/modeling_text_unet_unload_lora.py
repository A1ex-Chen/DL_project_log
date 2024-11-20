def unload_lora(self):
    """Unloads LoRA weights."""
    deprecate('unload_lora', '0.28.0',
        'Calling `unload_lora()` is deprecated and will be removed in a future version. Please install `peft` and then call `disable_adapters().'
        )
    for module in self.modules():
        if hasattr(module, 'set_lora_layer'):
            module.set_lora_layer(None)
