def load_depths(image_list, datadir, H=None, W=None):
    depths = []
    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, 'depth_{}.npy'.format(frame_id))
        depth = np.load(depth_path)
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)
    return np.stack(depths)
