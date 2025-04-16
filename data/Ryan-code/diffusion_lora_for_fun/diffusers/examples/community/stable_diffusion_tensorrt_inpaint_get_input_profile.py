def get_input_profile(self, batch_size, image_height, image_width,
    static_batch, static_shape):
    assert batch_size >= self.min_batch and batch_size <= self.max_batch
    min_batch = batch_size if static_batch else self.min_batch
    max_batch = batch_size if static_batch else self.max_batch
    self.check_dims(batch_size, image_height, image_width)
    (min_batch, max_batch, min_image_height, max_image_height,
        min_image_width, max_image_width, _, _, _, _) = (self.
        get_minmax_dims(batch_size, image_height, image_width, static_batch,
        static_shape))
    return {'images': [(min_batch, 3, min_image_height, min_image_width), (
        batch_size, 3, image_height, image_width), (max_batch, 3,
        max_image_height, max_image_width)]}
