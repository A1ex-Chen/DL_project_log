def _modify_text_encoder(self, attn_processors: Dict[str, LoRAAttnProcessor]):
    """
        Monkey-patches the forward passes of attention modules of the text encoder.

        Parameters:
            attn_processors: Dict[str, `LoRAAttnProcessor`]:
                A dictionary mapping the module names and their corresponding [`~LoRAAttnProcessor`].
        """
    self._remove_text_encoder_monkey_patch()
    for name, attn_module in self.text_encoder.named_modules():
        if name.endswith(TEXT_ENCODER_ATTN_MODULE):
            for attn_proc_attr, text_encoder_attr in self._lora_attn_processor_attr_to_text_encoder_attr.items(
                ):
                module = attn_module.get_submodule(text_encoder_attr)
                lora_layer = attn_processors[name].get_submodule(attn_proc_attr
                    )
                old_forward = module.old_forward = module.forward

                def make_new_forward(old_forward, lora_layer):

                    def new_forward(x):
                        result = old_forward(x) + self.lora_scale * lora_layer(
                            x)
                        return result
                    return new_forward
                module.forward = make_new_forward(old_forward, lora_layer)
