def run_safety_checker(self, image, device, dtype):
    feature_extractor_input = self.image_processor.postprocess(image,
        output_type='pil')
    safety_checker_input = self.feature_extractor(feature_extractor_input,
        return_tensors='pt').to(device)
    image, has_nsfw_concept = self.safety_checker(images=image, clip_input=
        safety_checker_input.pixel_values.to(dtype))
    return image, has_nsfw_concept
