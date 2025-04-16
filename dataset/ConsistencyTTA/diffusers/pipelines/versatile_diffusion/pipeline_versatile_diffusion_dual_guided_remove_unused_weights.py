def remove_unused_weights(self):
    self.register_modules(text_unet=None)
