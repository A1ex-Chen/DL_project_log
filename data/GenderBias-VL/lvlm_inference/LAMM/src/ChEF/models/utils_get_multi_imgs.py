def get_multi_imgs(images):
    imgs = []
    for image in images:
        imgs.append(get_image(image))
    return imgs
