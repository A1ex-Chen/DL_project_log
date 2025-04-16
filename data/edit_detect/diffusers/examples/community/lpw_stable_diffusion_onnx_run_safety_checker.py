def run_safety_checker(self, image):
    if self.safety_checker is not None:
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(
            image), return_tensors='np').pixel_values.astype(image.dtype)
        images, has_nsfw_concept = [], []
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(clip_input=
                safety_checker_input[i:i + 1], images=image[i:i + 1])
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    else:
        has_nsfw_concept = None
    return image, has_nsfw_concept
