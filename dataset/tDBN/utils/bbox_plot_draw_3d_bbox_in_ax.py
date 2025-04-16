def draw_3d_bbox_in_ax(ax, bboxes, colors='r', alpha=0.5, image_shape=None):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        box_a, box_b = box[:4], box[4:]
        for pa, pb in zip(box_a, box_a[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_b, box_b[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        for pa, pb in zip(box_a, box_b):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
    if image_shape is not None:
        patch = patches.Rectangle([0, 0], image_shape[1], image_shape[0])
        ax.set_clip_path(patch)
    return ax
