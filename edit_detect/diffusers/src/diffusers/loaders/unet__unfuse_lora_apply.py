def _unfuse_lora_apply(self, module):
    if not USE_PEFT_BACKEND:
        if hasattr(module, '_unfuse_lora'):
            module._unfuse_lora()
    else:
        from peft.tuners.tuners_utils import BaseTunerLayer
        if isinstance(module, BaseTunerLayer):
            module.unmerge()
