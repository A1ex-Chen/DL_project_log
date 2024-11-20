def crop(self, im, new_width, new_height):
    width, height = im.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return im.crop((left, top, right, bottom))
