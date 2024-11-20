def __init__(self):
    super().__init__()
    self.register_buffer('watermark_image', torch.zeros((62, 62, 4)))
    self.watermark_image_as_pil = None
