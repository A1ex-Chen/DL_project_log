def build_input_image(self, image_list):
    images = self.get_image_list(image_list)
    images = process_images(images, self.image_processor)
    images = images.to(self.device, dtype=torch.float16)
    return images
