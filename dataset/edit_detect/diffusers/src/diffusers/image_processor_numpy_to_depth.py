def numpy_to_depth(self, images: np.ndarray) ->List[PIL.Image.Image]:
    """
        Convert a NumPy depth image or a batch of images to a PIL image.
        """
    if images.ndim == 3:
        images = images[None, ...]
    images_depth = images[:, :, :, 3:]
    if images.shape[-1] == 6:
        images_depth = (images_depth * 255).round().astype('uint8')
        pil_images = [Image.fromarray(self.rgblike_to_depthmap(image_depth),
            mode='I;16') for image_depth in images_depth]
    elif images.shape[-1] == 4:
        images_depth = (images_depth * 65535.0).astype(np.uint16)
        pil_images = [Image.fromarray(image_depth, mode='I;16') for
            image_depth in images_depth]
    else:
        raise Exception('Not supported')
    return pil_images
