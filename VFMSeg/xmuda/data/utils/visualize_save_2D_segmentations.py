def save_2D_segmentations(org_image, anns, path=None, points=None,
    img_indices=None, batch_idx=0, org_labels=None, class_num=6,
    transparency=0.8, point_size=0.8):
    if len(anns) == 0:
        return
    org_image = org_image.permute(1, 2, 0)
    plt.figure(figsize=(20, 35))
    plt.imshow(org_image)
    if org_labels is not None:
        colors = []
        color_points = [[1, 0.62, 0.001, 0.9], [0.001, 0.81, 0.6, 0.9], [
            0.29, 0.001, 0.29, 0.9], [0.44, 0.71, 0.24, 0.9], [0.95, 0.001,
            0.001, 0.9], [0.001, 0.69, 0.001, 0.9], [0.0, 0.0, 0.0, 0.35]]
        for label in org_labels:
            if label == -100:
                colors.append(color_points[-1])
            else:
                colors.append(color_points[label])
        plt.scatter(img_indices[batch_idx][:, 1], img_indices[batch_idx][:,
            0], c=colors, alpha=1, s=1)
    elif points is not None and img_indices is not None:
        depth = points[:, 2]
        depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
        colors = []
        for depth_val in depth:
            colors.append(interpolate_or_clip(colormap=turbo_colormap_data,
                x=depth_val))
        plt.scatter(img_indices[batch_idx][:, 1], img_indices[batch_idx][:,
            0], c=colors, alpha=transparency, s=point_size)
    else:
        pass
    ax = plt.gca()
    img_mask = np.ones((anns.shape[0], anns.shape[1], 4))
    img_mask[:, :, 3] = 0
    color_masks = [[1, 0.62, 0.001, 0.2], [0.001, 0.81, 0.6, 0.2], [0.29, 
        0.001, 0.29, 0.2], [0.44, 0.71, 0.24, 0.2], [0.95, 0.001, 0.001, 
        0.2], [0.001, 0.69, 0.001, 0.2], [0.1, 0.9, 0.3, 0.35], [0.2, 0.7, 
        0.4, 0.35], [0.3, 0.5, 0.5, 0.35], [0.4, 0.4, 0.6, 0.35], [0.5, 0.3,
        0.7, 0.35], [0.6, 0.2, 0.8, 0.35], [0.7, 0.8, 0.6, 0.35], [0.8, 0.3,
        0.4, 0.35], [0.3, 0.3, 0.9, 0.35], [0.0, 0.0, 0.0, 0.35]]
    for i in range(0, class_num):
        mask = anns == i
        img_mask[mask] = color_masks[i]
    ax.imshow(img_mask)
    plt.axis('off')
    if path is not None:
        plt.savefig(path)
