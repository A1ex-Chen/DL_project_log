def set_image(self, img, switch):
    if img.mode not in ['RGB', 'L']:
        img = img.convert('RGB')
    if switch:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((512, 512), Image.BILINEAR)
    return img
