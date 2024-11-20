def ssim(img1, img2, use_padding=True, window_size=11, size_average=True):
    """SSIM only defined at intensity channel. For RGB or YUV or other image format, this function computes SSIm at each
    channel and averge them.
    :param img1:  (B, C, H, W)  float32 in [0, 1]
    :param img2:  (B, C, H, W)  float32 in [0, 1]
    :param use_padding: we use conv2d when we compute mean and var for each patch, this use_padding is for that conv2d.
    :param window_size: patch size
    :param size_average:
    :return:  a tensor that contains only one scalar.
    """
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, use_padding,
        size_average)
