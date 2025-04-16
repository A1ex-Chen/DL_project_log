def __init__(self, *args, lora_layer: Optional[LoRALinearLayer]=None, **kwargs
    ):
    deprecation_message = (
        'Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.'
        )
    deprecate('LoRACompatibleLinear', '1.0.0', deprecation_message)
    super().__init__(*args, **kwargs)
    self.lora_layer = lora_layer
