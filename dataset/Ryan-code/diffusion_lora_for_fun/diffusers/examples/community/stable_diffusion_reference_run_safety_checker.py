def run_safety_checker(self, image: Union[torch.Tensor, PIL.Image.Image],
    device: torch.device, dtype: torch.dtype) ->Tuple[Union[torch.Tensor,
    PIL.Image.Image], Optional[bool]]:
    """
        Runs the safety checker on the given image.

        Args:
            image (Union[torch.Tensor, PIL.Image.Image]): The input image to be checked.
            device (torch.device): The device to run the safety checker on.
            dtype (torch.dtype): The data type of the input image.

        Returns:
            (image, has_nsfw_concept) Tuple[Union[torch.Tensor, PIL.Image.Image], Optional[bool]]: A tuple containing the processed image and
            a boolean indicating whether the image has a NSFW (Not Safe for Work) concept.
        """
    if self.safety_checker is None:
        has_nsfw_concept = None
    else:
        if torch.is_tensor(image):
            feature_extractor_input = self.image_processor.postprocess(image,
                output_type='pil')
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)
        safety_checker_input = self.feature_extractor(feature_extractor_input,
            return_tensors='pt').to(device)
        image, has_nsfw_concept = self.safety_checker(images=image,
            clip_input=safety_checker_input.pixel_values.to(dtype))
    return image, has_nsfw_concept
