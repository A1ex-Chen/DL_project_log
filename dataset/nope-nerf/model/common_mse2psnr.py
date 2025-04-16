def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)
