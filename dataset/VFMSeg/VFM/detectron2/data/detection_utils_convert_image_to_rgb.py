def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.

    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == 'BGR':
        image = image[:, :, [2, 1, 0]]
    elif format == 'YUV-BT.601':
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == 'L':
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert('RGB'))
    return image
