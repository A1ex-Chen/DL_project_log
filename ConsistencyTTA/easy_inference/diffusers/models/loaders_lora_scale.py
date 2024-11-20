@property
def lora_scale(self) ->float:
    return self._lora_scale if hasattr(self, '_lora_scale') else 1.0
