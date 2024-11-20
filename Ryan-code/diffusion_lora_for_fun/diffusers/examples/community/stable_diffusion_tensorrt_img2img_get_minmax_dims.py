def get_minmax_dims(self, batch_size, image_height, image_width,
    static_batch, static_shape):
    min_batch = batch_size if static_batch else self.min_batch
    max_batch = batch_size if static_batch else self.max_batch
    latent_height = image_height // 8
    latent_width = image_width // 8
    min_image_height = image_height if static_shape else self.min_image_shape
    max_image_height = image_height if static_shape else self.max_image_shape
    min_image_width = image_width if static_shape else self.min_image_shape
    max_image_width = image_width if static_shape else self.max_image_shape
    min_latent_height = (latent_height if static_shape else self.
        min_latent_shape)
    max_latent_height = (latent_height if static_shape else self.
        max_latent_shape)
    min_latent_width = latent_width if static_shape else self.min_latent_shape
    max_latent_width = latent_width if static_shape else self.max_latent_shape
    return (min_batch, max_batch, min_image_height, max_image_height,
        min_image_width, max_image_width, min_latent_height,
        max_latent_height, min_latent_width, max_latent_width)
