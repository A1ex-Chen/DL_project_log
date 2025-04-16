def box3d_to_bbox(box3d, rect, Trv2c, P2):
    box_corners = center_to_corner_box3d(box3d[:, :3], box3d[:, 3:6], box3d
        [:, 6], [0.5, 1.0, 0.5], axis=1)
    box_corners_in_image = project_to_image(box_corners, P2)
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox
