@staticmethod
def numpy_to_pil(images):
    """
        Convert a numpy image or a batch of images to a PIL image.
        """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype('uint8')
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode='L') for image in
            images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images
