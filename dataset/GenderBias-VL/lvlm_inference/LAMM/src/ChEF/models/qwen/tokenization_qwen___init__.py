def __init__(self, img_rgb, metadata=None, scale=1.0):
    self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
    self.font_path = FONT_PATH
    self.output = VisImage(self.img, scale=scale)
    self.cpu_device = torch.device('cpu')
    self._default_font_size = max(np.sqrt(self.output.height * self.output.
        width) // 30, 15 // scale)
