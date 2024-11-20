def select_images(imgpoints, objpoints, points_to_use: int):
    """Select a subset of images based on their location in the image plane.

    This function selects a subset of images based on the location of the corners in the image plane. The images are sorted
    by their distance to the origin and a specified number of images is selected to use.

    Args:
        imgpoints (list): A list of 2D points in the image plane.
        objpoints (list): A list of 3D points in real world space.
        points_to_use (int): The number of images to use. Must be greater than 0.

    Returns:
        tuple: A tuple containing two lists, `objpoints` and `imgpoints`. `objpoints` is a list of 3D points in real world space, and `imgpoints` is a list of 2D points in the image plane.

    Raises:
        ValueError: If `points_to_use` is less than or equal to 0.
    """
    if points_to_use <= 0:
        raise ValueError('The number of points to use must be greater than 0.')
    if len(imgpoints) <= points_to_use:
        return imgpoints, objpoints
    X = np.asarray([np.ravel(x) for x in imgpoints])
    pca = PCA(n_components=1)
    Xt = np.ravel(pca.fit_transform(X))
    idxs = np.argsort(Xt)
    objpoints = [objpoints[i] for i in idxs]
    imgpoints = [imgpoints[i] for i in idxs]
    x_range = np.linspace(0, len(imgpoints) - 1, points_to_use, endpoint=
        False, dtype=int)
    objpoints = [objpoints[i] for i in x_range]
    imgpoints = [imgpoints[i] for i in x_range]
    return imgpoints, objpoints
