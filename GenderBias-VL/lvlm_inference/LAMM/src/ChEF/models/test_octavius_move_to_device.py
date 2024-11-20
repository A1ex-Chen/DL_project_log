def move_to_device(self):
    if torch.cuda.is_available():
        self.dtype = torch.float16
        self.device = 'cuda'
    else:
        self.dtype = torch.float32
        self.device = 'cpu'
    self.model = self.model.to(self.device, dtype=self.dtype)
