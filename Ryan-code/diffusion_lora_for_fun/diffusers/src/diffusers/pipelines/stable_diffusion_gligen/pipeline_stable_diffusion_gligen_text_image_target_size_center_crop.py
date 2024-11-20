def target_size_center_crop(self, im, new_hw):
    """
        Crop and resize the image to the target size while keeping the center.
        """
    width, height = im.size
    if width != height:
        im = self.crop(im, min(height, width), min(height, width))
    return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)
