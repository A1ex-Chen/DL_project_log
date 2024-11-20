def build_input_image(self, image_list):
    image_list = self.get_image_list(image_list)
    image_tensors = []
    for image in image_list:
        image_tensor = self.vis_processors['eval'](image).unsqueeze(0).to(self
            .device)
        image_tensors.append(image_tensor)
    image_tensors = torch.cat(image_tensors)
    image_tensors = image_tensors.permute(1, 0, 2, 3)
    return image_tensors
