@staticmethod
def depth_pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]
    ) ->np.ndarray:
    """
        Convert a PIL image or a list of PIL images to NumPy arrays.
        """
    if not isinstance(images, list):
        images = [images]
    images = [(np.array(image).astype(np.float32) / (2 ** 16 - 1)) for
        image in images]
    images = np.stack(images, axis=0)
    return images
