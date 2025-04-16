def undistort_image(config, image, cam_name):
    """copied from https://www.a2d2.audi/a2d2/en/tutorial.html"""
    if cam_name in ['front_left', 'front_center', 'front_right',
        'side_left', 'side_right', 'rear_center']:
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = np.asarray(config['cameras'][cam_name][
            'CamMatrixOriginal'])
        dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=
                dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=
                dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image
