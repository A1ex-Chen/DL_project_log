def run_safety_checker(self, image, device, dtype):
    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='pt').to(device)
        image, has_nsfw_concept = self.safety_checker(images=image,
            clip_input=safety_checker_input.pixel_values.to(dtype))
    else:
        has_nsfw_concept = None
    return image, has_nsfw_concept
