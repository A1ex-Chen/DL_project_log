def check_dims(self, batch_size, image_height, image_width):
    assert batch_size >= self.min_batch and batch_size <= self.max_batch
    assert image_height % 8 == 0 or image_width % 8 == 0
    latent_height = image_height // 8
    latent_width = image_width // 8
    assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
    assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
    return latent_height, latent_width
