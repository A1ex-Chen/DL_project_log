def __init__(self):
    self.watermark = WATERMARK_BITS
    self.encoder = WatermarkEncoder()
    self.encoder.set_watermark('bits', self.watermark)
