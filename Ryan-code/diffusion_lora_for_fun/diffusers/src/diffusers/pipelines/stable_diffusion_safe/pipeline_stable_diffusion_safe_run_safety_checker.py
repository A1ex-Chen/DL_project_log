def run_safety_checker(self, image, device, dtype, enable_safety_guidance):
    if self.safety_checker is not None:
        images = image.copy()
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='pt').to(device)
        image, has_nsfw_concept = self.safety_checker(images=image,
            clip_input=safety_checker_input.pixel_values.to(dtype))
        flagged_images = np.zeros((2, *image.shape[1:]))
        if any(has_nsfw_concept):
            logger.warning(
                f"Potential NSFW content was detected in one or more images. A black image will be returned instead.{'You may look at this images in the `unsafe_images` variable of the output at your own discretion.' if enable_safety_guidance else 'Try again with a different prompt and/or seed.'}"
                )
            for idx, has_nsfw_concept in enumerate(has_nsfw_concept):
                if has_nsfw_concept:
                    flagged_images[idx] = images[idx]
                    image[idx] = np.zeros(image[idx].shape)
    else:
        has_nsfw_concept = None
        flagged_images = None
    return image, has_nsfw_concept, flagged_images
