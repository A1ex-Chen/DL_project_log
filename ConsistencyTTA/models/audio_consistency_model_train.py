def train(self, mode: bool=True):
    super().train(mode)
    self.student_target_unet.eval()
    self.vae.eval()
    return self
