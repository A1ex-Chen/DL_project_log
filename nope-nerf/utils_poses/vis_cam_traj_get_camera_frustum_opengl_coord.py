def get_camera_frustum_opengl_coord(H, W, fx, fy, W2C, frustum_length=0.5,
    color=np.array([0.0, 1.0, 0.0])):
    """X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    """
    hfov = np.rad2deg(np.arctan(W / 2.0 / fx) * 2.0)
    vfov = np.rad2deg(np.arctan(H / 2.0 / fy) * 2.0)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.0))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.0))
    frustum_points = np.array([[0.0, 0.0, 0.0, 1.0], [-half_w, half_h, -
        frustum_length, 1.0], [half_w, half_h, -frustum_length, 1.0], [
        half_w, -half_h, -frustum_length, 1.0], [-half_w, -half_h, -
        frustum_length, 1.0]])
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, i + 1] for
        i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(color.reshape((1, 3)), (frustum_lines.shape[0], 1)
        )
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
    return frustum_points, frustum_lines, frustum_colors
