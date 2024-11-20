def load_depths_npz(image_list, datadir, H=None, W=None, norm=False):
    depths = []
    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, 'depth_{}.npz'.format(frame_id))
        depth = np.load(depth_path)['pred']
        if depth.shape[0] == 1:
            depth = depth[0]
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)
    depths = np.stack(depths)
    if norm:
        depths_n = []
        t_all = np.median(depths)
        s_all = np.mean(np.abs(depths - t_all))
        for depth in depths:
            t_i = np.median(depth)
            s_i = np.mean(np.abs(depth - t_i))
            depth = s_all * (depth - t_i) / s_i + t_all
            depths_n.append(depth)
        depths = np.stack(depths_n)
    return depths
