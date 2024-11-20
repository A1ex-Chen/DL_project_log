def image_to_tensor(self, image: Union[str, Image.Image], dim: tuple, dtype):
    """
        Convert latent PIL image to a torch tensor for further processing.
        """
    if isinstance(image, str):
        image = Image.open(image)
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    image = self.image_processor.preprocess(image, height=dim[0], width=dim[1]
        )[0]
    return image.type(dtype)
