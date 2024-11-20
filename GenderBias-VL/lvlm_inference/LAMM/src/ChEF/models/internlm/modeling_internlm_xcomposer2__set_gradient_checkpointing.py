def _set_gradient_checkpointing(self, module, value=False):
    if isinstance(module, InternLM2Model):
        module.gradient_checkpointing = value
    if value:
        (self.vit.vision_tower.vision_model.encoder.gradient_checkpointing
            ) = value
