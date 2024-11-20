def _caffe2_preprocess_image(self, inputs):
    """
        Caffe2 implementation of preprocess_image, which is called inside each MetaArch's forward.
        It normalizes the input images, and the final caffe2 graph assumes the
        inputs have been batched already.
        """
    data, im_info = inputs
    data = alias(data, 'data')
    im_info = alias(im_info, 'im_info')
    mean, std = self._wrapped_model.pixel_mean, self._wrapped_model.pixel_std
    normalized_data = (data - mean) / std
    normalized_data = alias(normalized_data, 'normalized_data')
    images = ImageList(tensor=normalized_data, image_sizes=im_info)
    return images
