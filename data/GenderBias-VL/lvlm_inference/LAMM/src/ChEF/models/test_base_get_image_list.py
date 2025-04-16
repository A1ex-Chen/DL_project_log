def get_image_list(self, image_list):
    if not isinstance(image_list, list):
        image_list = [image_list]
    return get_multi_imgs(image_list)
