def build_input_image(self, image_list):
    image_list = self.get_image_list(image_list)
    if len(image_list) == 1:
        image = image_list[0]
    else:
        image = self.horizontal_concat(image_list)
    return image
