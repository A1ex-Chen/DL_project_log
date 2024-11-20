def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if 'width' in dataset_dict or 'height' in dataset_dict:
        image_wh = image.shape[1], image.shape[0]
        expected_wh = dataset_dict['width'], dataset_dict['height']
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                'Mismatched image shape{}, got {}, expect {}.'.format(
                ' for image ' + dataset_dict['file_name'] if 'file_name' in
                dataset_dict else '', image_wh, expected_wh) +
                ' Please check the width/height in your annotation.')
    if 'width' not in dataset_dict:
        dataset_dict['width'] = image.shape[1]
    if 'height' not in dataset_dict:
        dataset_dict['height'] = image.shape[0]
