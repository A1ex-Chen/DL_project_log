def _remove_text_encoder_monkey_patch(self):
    for name, attn_module in self.text_encoder.named_modules():
        if name.endswith(TEXT_ENCODER_ATTN_MODULE):
            for _, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items(
                ):
                module = attn_module.get_submodule(text_encoder_attr)
                if hasattr(module, 'old_forward'):
                    module.forward = module.old_forward
                    delattr(module, 'old_forward')
