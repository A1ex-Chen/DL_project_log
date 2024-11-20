def freeze_unet(self):
    set_requires_grad(self.unet, False)
