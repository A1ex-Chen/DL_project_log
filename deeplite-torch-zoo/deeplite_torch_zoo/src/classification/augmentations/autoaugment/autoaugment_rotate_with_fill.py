def rotate_with_fill(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot
        ).convert(img.mode)
