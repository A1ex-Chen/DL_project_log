def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate
    ):
    new_height, new_width = new_shape
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2
    C[1, 2] = -img_shape[0] / 2
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height
    M = T @ S @ R @ C
    return M, s
