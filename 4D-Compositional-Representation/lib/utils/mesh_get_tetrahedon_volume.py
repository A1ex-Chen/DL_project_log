def get_tetrahedon_volume(points):
    vectors = points[..., :3, :] - points[..., 3:, :]
    volume = 1 / 6 * np.linalg.det(vectors)
    return volume
