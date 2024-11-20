def load_gt_depths(image_list, datadir, H=None, W=None, crop_ratio=1):
    depths = []
    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, 'depth', '{}.png'.format(frame_id))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        if crop_ratio != 1:
            h, w = depth.shape
            crop_size_h = int(h * crop_ratio)
            crop_size_w = int(w * crop_ratio)
            depth = depth[crop_size_h:h - crop_size_h, crop_size_w:w -
                crop_size_w]
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.
                INTER_NEAREST)
            depths.append(depth_resize)
        else:
            depths.append(depth)
    return np.stack(depths)
