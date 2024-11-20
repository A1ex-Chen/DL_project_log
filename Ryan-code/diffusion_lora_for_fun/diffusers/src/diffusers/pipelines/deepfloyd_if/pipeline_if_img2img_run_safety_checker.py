def run_safety_checker(self, image, device, dtype):
    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='pt').to(device)
        image, nsfw_detected, watermark_detected = self.safety_checker(images
            =image, clip_input=safety_checker_input.pixel_values.to(dtype=
            dtype))
    else:
        nsfw_detected = None
        watermark_detected = None
    return image, nsfw_detected, watermark_detected
