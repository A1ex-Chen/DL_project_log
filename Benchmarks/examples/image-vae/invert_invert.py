def invert(self, img):
    """Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        rgb = Image.merge('RGB', (r, g, b))
        inv = ImageOps.invert(rgb)
        r, g, b = inv.split()
        inv = Image.merge('RGBA', (r, g, b, a))
    elif img.mode == 'LA':
        m, a = img.split()
        m = ImageOps.invert(m)
        inv = Image.merge('LA', (m, a))
    else:
        inv = ImageOps.invert(img)
    return inv
