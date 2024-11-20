@staticmethod
def get_crop_region(mask_image: PIL.Image.Image, width: int, height: int, pad=0
    ):
    """
        Finds a rectangular region that contains all masked ares in an image, and expands region to match the aspect
        ratio of the original image; for example, if user drew mask in a 128x32 region, and the dimensions for
        processing are 512x512, the region will be expanded to 128x128.

        Args:
            mask_image (PIL.Image.Image): Mask image.
            width (int): Width of the image to be processed.
            height (int): Height of the image to be processed.
            pad (int, optional): Padding to be added to the crop region. Defaults to 0.

        Returns:
            tuple: (x1, y1, x2, y2) represent a rectangular region that contains all masked ares in an image and
            matches the original aspect ratio.
        """
    mask_image = mask_image.convert('L')
    mask = np.array(mask_image)
    h, w = mask.shape
    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1
    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1
    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1
    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1
    x1, y1, x2, y2 = int(max(crop_left - pad, 0)), int(max(crop_top - pad, 0)
        ), int(min(w - crop_right + pad, w)), int(min(h - crop_bottom + pad, h)
        )
    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = width / height
    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= mask_image.height:
            diff = y2 - mask_image.height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= mask_image.height:
            y2 = mask_image.height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= mask_image.width:
            diff = x2 - mask_image.width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= mask_image.width:
            x2 = mask_image.width
    return x1, y1, x2, y2
