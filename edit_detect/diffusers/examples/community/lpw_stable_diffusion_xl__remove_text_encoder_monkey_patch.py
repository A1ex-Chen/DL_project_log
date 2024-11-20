def _remove_text_encoder_monkey_patch(self):
    self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
    self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)
