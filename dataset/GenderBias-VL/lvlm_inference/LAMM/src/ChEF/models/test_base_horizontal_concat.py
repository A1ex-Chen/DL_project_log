def horizontal_concat(self, image_list):
    total_width = sum(img.width for img in image_list)
    max_height = max(img.height for img in image_list)
    dst = Image.new('RGB', (total_width, max_height))
    current_width = 0
    for img in image_list:
        dst.paste(img, (current_width, 0))
        current_width += img.width
    return dst
