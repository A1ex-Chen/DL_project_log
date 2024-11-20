def draw_points_image_labels(img, img_indices, seg_labels, show=True,
    color_palette_type='NuScenes', point_size=0.5):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSegLong':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'VirtualKITTI':
        color_palette = VIRTUAL_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.0
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5,
        s=point_size)
    plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
