def __init__(self, unet, controlnets: ControlNetModel):
    super().__init__()
    self.unet = unet
    self.controlnets = controlnets
