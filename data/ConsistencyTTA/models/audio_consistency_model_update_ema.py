def update_ema(self):
    assert self.training, 'EMA update should only be called during training'
    do_ema_update(source_model=self.student_unet, shadow_models=[self.
        student_target_unet, self.student_ema_unet], decay_consts=[self.
        target_ema_decay, self.ema_decay])
