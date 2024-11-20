def _remove_text_encoder_monkey_patch(self):
    recurse_remove_peft_layers(self.text_encoder)
    if getattr(self.text_encoder, 'peft_config', None) is not None:
        del self.text_encoder.peft_config
        self.text_encoder._hf_peft_config_loaded = None
    recurse_remove_peft_layers(self.text_encoder_2)
    if getattr(self.text_encoder_2, 'peft_config', None) is not None:
        del self.text_encoder_2.peft_config
        self.text_encoder_2._hf_peft_config_loaded = None
