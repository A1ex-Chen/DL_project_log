def __preprocess_images(self, batch_size, images=()):
    init_images = []
    for image in images:
        image = image.to(self.torch_device).float()
        image = image.repeat(batch_size, 1, 1, 1)
        init_images.append(image)
    return tuple(init_images)
