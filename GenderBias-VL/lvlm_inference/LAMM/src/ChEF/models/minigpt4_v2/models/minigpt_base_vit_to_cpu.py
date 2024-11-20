def vit_to_cpu(self):
    self.ln_vision.to('cpu')
    self.ln_vision.float()
    self.visual_encoder.to('cpu')
    self.visual_encoder.float()
