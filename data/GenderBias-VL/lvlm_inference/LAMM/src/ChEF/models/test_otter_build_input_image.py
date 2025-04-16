def build_input_image(self, image_list):
    imgs = self.get_image_list(image_list)
    imgs = self.image_processor.preprocess(imgs, return_tensors='pt')[
        'pixel_values'].unsqueeze(1)
    return imgs
