def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = 3, 3
    else:
        L = np.arange(-radius, radius + 1)
        ksize = 5, 5
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array(X ** 2 + Y ** 2 <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
