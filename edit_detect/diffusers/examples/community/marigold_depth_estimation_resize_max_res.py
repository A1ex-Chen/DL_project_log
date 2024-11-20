@staticmethod
def resize_max_res(img: Image.Image, max_edge_resolution: int,
    resample_method=Resampling.BILINEAR) ->Image.Image:
    """
        Resize image to limit maximum edge length while keeping aspect ratio.

        Args:
            img (`Image.Image`):
                Image to be resized.
            max_edge_resolution (`int`):
                Maximum edge length (pixel).
            resample_method (`PIL.Image.Resampling`):
                Resampling method used to resize images.

        Returns:
            `Image.Image`: Resized image.
        """
    original_width, original_height = img.size
    downscale_factor = min(max_edge_resolution / original_width, 
        max_edge_resolution / original_height)
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    resized_img = img.resize((new_width, new_height), resample=resample_method)
    return resized_img
