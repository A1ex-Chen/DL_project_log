def build_input_image(self, image_list):
    images = self.get_image_list(image_list)
    image_tensor = []
    for image in images:
        image_tensor.append(self.vis_processor(image).unsqueeze(0).to(self.
            device))
    return image_tensor
