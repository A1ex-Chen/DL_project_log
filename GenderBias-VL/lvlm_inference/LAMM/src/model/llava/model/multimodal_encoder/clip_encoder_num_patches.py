@property
def num_patches(self):
    return (self.config.image_size // self.config.patch_size) ** 2
