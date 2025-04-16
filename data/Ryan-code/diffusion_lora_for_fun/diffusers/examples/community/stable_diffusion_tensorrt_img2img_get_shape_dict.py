def get_shape_dict(self, batch_size, image_height, image_width):
    latent_height, latent_width = self.check_dims(batch_size, image_height,
        image_width)
    return {'images': (batch_size, 3, image_height, image_width), 'latent':
        (batch_size, 4, latent_height, latent_width)}
