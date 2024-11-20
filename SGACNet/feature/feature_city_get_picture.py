def get_picture(picture_dir, transform):
    """
    该算法实现了读取图片，并将其类型转化为Tensor
    """
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (480, 640))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)
    return transform(img256)
