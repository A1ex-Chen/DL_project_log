def rotate_func(img, degree, fill=(0, 0, 0)):
    """
    like PIL, rotate by degree, not radians
    """
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out
