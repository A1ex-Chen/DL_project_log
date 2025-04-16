def build_input_image(self, image_list):
    images = self.get_image_list(image_list)
    images = [self.image_processor(image, return_tensors='pt').pixel_values for
        image in images]
    images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
    return images
