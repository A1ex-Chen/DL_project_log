def build_input_image(self, image_list):
    image_list = self.get_image_list(image_list)
    image_tensors = []
    for image in image_list:
        image_tensor = self.image_processor.preprocess(image,
            return_tensors='pt')['pixel_values'].half().to(self.device)
        image_tensors.append(image_tensor)
    image_tensors = torch.cat(image_tensors)
    return image_tensors
