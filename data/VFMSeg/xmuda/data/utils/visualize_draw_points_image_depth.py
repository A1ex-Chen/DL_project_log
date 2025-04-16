def draw_points_image_depth(img, img_indices, depth, show=True, point_size=0.5
    ):
    depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    colors = []
    for depth_val in depth:
        colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=
            depth_val))
    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5,
        s=point_size)
    plt.axis('off')
    if show:
        plt.show()
