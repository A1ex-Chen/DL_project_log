def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor ** 2
    if height % scale_factor ** 2 != 0:
        new_height += 1
    new_width = width // scale_factor ** 2
    if width % scale_factor ** 2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor
