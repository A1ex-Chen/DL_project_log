def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=
            PIL_INTERPOLATION['lanczos'])) for i in image]
        image = [(i[None, ..., None] if i.ndim == 2 else i[None, ...]) for
            i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f'Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}'
                )
    return image
