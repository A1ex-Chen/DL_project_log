def get_input_profile(self, batch_size, image_height, image_width,
    static_batch, static_shape):
    latent_height, latent_width = self.check_dims(batch_size, image_height,
        image_width)
    (min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height,
        min_latent_width, max_latent_width) = (self.get_minmax_dims(
        batch_size, image_height, image_width, static_batch, static_shape))
    return {'latent': [(min_batch, 4, min_latent_height, min_latent_width),
        (batch_size, 4, latent_height, latent_width), (max_batch, 4,
        max_latent_height, max_latent_width)]}
