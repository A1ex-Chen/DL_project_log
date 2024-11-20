def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
    deprecation_message = (
        'Use of `set_lora_layer()` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.'
        )
    deprecate('set_lora_layer', '1.0.0', deprecation_message)
    self.lora_layer = lora_layer
