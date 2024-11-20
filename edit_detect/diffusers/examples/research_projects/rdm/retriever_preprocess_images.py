def preprocess_images(images: List[np.array], feature_extractor:
    CLIPFeatureExtractor) ->torch.Tensor:
    """
    Preprocesses a list of images into a batch of tensors.

    Args:
        images (:obj:`List[Image.Image]`):
            A list of images to preprocess.

    Returns:
        :obj:`torch.Tensor`: A batch of tensors.
    """
    images = [np.array(image) for image in images]
    images = [((image + 1.0) / 2.0) for image in images]
    images = feature_extractor(images, return_tensors='pt').pixel_values
    return images
