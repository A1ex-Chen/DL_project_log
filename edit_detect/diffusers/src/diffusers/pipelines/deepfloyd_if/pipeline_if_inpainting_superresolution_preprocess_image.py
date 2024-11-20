def preprocess_image(self, image: PIL.Image.Image, num_images_per_prompt,
    device) ->torch.Tensor:
    if not isinstance(image, torch.Tensor) and not isinstance(image, list):
        image = [image]
    if isinstance(image[0], PIL.Image.Image):
        image = [(np.array(i).astype(np.float32) / 127.5 - 1.0) for i in image]
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image.transpose(0, 3, 1, 2))
    elif isinstance(image[0], np.ndarray):
        image = np.stack(image, axis=0)
        if image.ndim == 5:
            image = image[0]
        image = torch.from_numpy(image.transpose(0, 3, 1, 2))
    elif isinstance(image, list) and isinstance(image[0], torch.Tensor):
        dims = image[0].ndim
        if dims == 3:
            image = torch.stack(image, dim=0)
        elif dims == 4:
            image = torch.concat(image, dim=0)
        else:
            raise ValueError(
                f'Image must have 3 or 4 dimensions, instead got {dims}')
    image = image.to(device=device, dtype=self.unet.dtype)
    image = image.repeat_interleave(num_images_per_prompt, dim=0)
    return image
