def freeze_vae(self):
    set_requires_grad(self.vae, False)
