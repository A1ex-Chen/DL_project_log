def move_to_device(self, device):
    self.dtype = torch.float16
    self.device = device
    convert_weights_to_fp16(self.model.vision_encoder)
    self.model = self.model.to(self.device, dtype=self.dtype)
