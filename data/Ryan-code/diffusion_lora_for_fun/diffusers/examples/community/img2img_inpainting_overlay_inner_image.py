def overlay_inner_image(image, inner_image, paste_offset: Tuple[int]=(0, 0)):
    inner_image = inner_image.convert('RGBA')
    image = image.convert('RGB')
    image.paste(inner_image, paste_offset, inner_image)
    image = image.convert('RGB')
    return image
