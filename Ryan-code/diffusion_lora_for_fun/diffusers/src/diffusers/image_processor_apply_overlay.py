def apply_overlay(self, mask: PIL.Image.Image, init_image: PIL.Image.Image,
    image: PIL.Image.Image, crop_coords: Optional[Tuple[int, int, int, int]
    ]=None) ->PIL.Image.Image:
    """
        overlay the inpaint output to the original image
        """
    width, height = image.width, image.height
    init_image = self.resize(init_image, width=width, height=height)
    mask = self.resize(mask, width=width, height=height)
    init_image_masked = PIL.Image.new('RGBa', (width, height))
    init_image_masked.paste(init_image.convert('RGBA').convert('RGBa'),
        mask=ImageOps.invert(mask.convert('L')))
    init_image_masked = init_image_masked.convert('RGBA')
    if crop_coords is not None:
        x, y, x2, y2 = crop_coords
        w = x2 - x
        h = y2 - y
        base_image = PIL.Image.new('RGBA', (width, height))
        image = self.resize(image, height=h, width=w, resize_mode='crop')
        base_image.paste(image, (x, y))
        image = base_image.convert('RGB')
    image = image.convert('RGBA')
    image.alpha_composite(init_image_masked)
    image = image.convert('RGB')
    return image
