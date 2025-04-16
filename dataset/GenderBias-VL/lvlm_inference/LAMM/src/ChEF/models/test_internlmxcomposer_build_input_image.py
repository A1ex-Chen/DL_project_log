def build_input_image(self, image_list):
    image_list = self.get_image_list(image_list)
    if len(image_list) >= 4:
        image_list = [self.horizontal_concat(image_list)]
    image_tensors = []
    for image in image_list:
        image_tensor = self.model.vis_processor(image).unsqueeze(0).to(self
            .device)
        image_tensors.append(image_tensor)
    image_tensors = torch.cat(image_tensors)
    return image_tensors
