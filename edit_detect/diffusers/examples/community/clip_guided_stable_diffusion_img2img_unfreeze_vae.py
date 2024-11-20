def unfreeze_vae(self):
    set_requires_grad(self.vae, True)
