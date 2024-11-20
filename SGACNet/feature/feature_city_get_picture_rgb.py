def get_picture_rgb(picture_dir):
    """
    该函数实现了显示图片的RGB三通道颜色
    """
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (960, 1280))
    skimage.io.imsave('0058.png', img256)
    img = img256.copy()
    axs = plt.subplot()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
