def unfreeze_unet(self):
    set_requires_grad(self.unet, True)
