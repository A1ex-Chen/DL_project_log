def prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]]):
    if not isinstance(image, (Image.Image, list)):
        raise ValueError(
            f'image has to be of type `PIL.Image.Image` or list but is {type(image)}'
            )
    if isinstance(image, Image.Image):
        image = [image]
    processed_images = jnp.concatenate([preprocess(img, jnp.float32) for
        img in image])
    return processed_images
