def train(self, mode: bool=True):
    super().train(mode)
    if self.freeze_text_encoder:
        self.text_encoder.eval()
    self.teacher_unet.eval()
    self.student_ema_unet.eval()
    return self
