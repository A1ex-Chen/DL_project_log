def preprocess_original_image(self, image: PIL.Image.Image) ->torch.Tensor:
    if not isinstance(image, list):
        image = [image]

    def numpy_to_pt(images):
        if images.ndim == 3:
            images = images[..., None]
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images
    if isinstance(image[0], PIL.Image.Image):
        new_image = []
        for image_ in image:
            image_ = image_.convert('RGB')
            image_ = resize(image_, self.unet.config.sample_size)
            image_ = np.array(image_)
            image_ = image_.astype(np.float32)
            image_ = image_ / 127.5 - 1
            new_image.append(image_)
        image = new_image
        image = np.stack(image, axis=0)
        image = numpy_to_pt(image)
    elif isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0) if image[0
            ].ndim == 4 else np.stack(image, axis=0)
        image = numpy_to_pt(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, axis=0) if image[0
            ].ndim == 4 else torch.stack(image, axis=0)
    return image
