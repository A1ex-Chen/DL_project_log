def _run_safety_checker(self, images, safety_model_params, jit=False):
    pil_images = [Image.fromarray(image) for image in images]
    features = self.feature_extractor(pil_images, return_tensors='np'
        ).pixel_values
    if jit:
        features = shard(features)
        has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features,
            safety_model_params)
        has_nsfw_concepts = unshard(has_nsfw_concepts)
        safety_model_params = unreplicate(safety_model_params)
    else:
        has_nsfw_concepts = self._get_has_nsfw_concepts(features,
            safety_model_params)
    images_was_copied = False
    for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        if has_nsfw_concept:
            if not images_was_copied:
                images_was_copied = True
                images = images.copy()
            images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)
        if any(has_nsfw_concepts):
            warnings.warn(
                'Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.'
                )
    return images, has_nsfw_concepts
