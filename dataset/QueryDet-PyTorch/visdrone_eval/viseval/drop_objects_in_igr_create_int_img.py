def create_int_img(img):
    int_img = np.cumsum(img, axis=0)
    np.cumsum(int_img, axis=1, out=int_img)
    return int_img
