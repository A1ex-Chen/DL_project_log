def draw_camera_frustum_geometry(c2ws, H, W, fx=600.0, fy=600.0,
    frustum_length=0.5, color=np.array([29.0, 53.0, 87.0]) / 255.0,
    draw_now=False, coord='opengl'):
    """
    :param c2ws:            (N, 4, 4)  np.array
    :param H:               scalar
    :param W:               scalar
    :param fx:              scalar
    :param fy:              scalar
    :param frustum_length:  scalar
    :param color:           None or (N, 3) or (3, ) or (1, 3) or (3, 1) np array
    :param draw_now:        True/False call o3d vis now
    :return:
    """
    N = c2ws.shape[0]
    num_ele = color.flatten().shape[0]
    if num_ele == 3:
        color = color.reshape(1, 3)
        color = np.tile(color, (N, 1))
    frustum_list = []
    if coord == 'opengl':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opengl_coord(H, W, fx,
                fy, W2C=np.linalg.inv(c2ws[i]), frustum_length=
                frustum_length, color=color[i]))
    elif coord == 'opencv':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opencv_coord(H, W, fx,
                fy, W2C=np.linalg.inv(c2ws[i]), frustum_length=
                frustum_length, color=color[i]))
    else:
        print('Undefined coordinate system. Exit')
        exit()
    frustums_geometry = frustums2lineset(frustum_list)
    if draw_now:
        o3d.visualization.draw_geometries([frustums_geometry])
    return frustums_geometry
