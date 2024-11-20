def draw_2d_bbox_in_ax(ax, bboxes, colors='r', alpha=0.5, with_arrow=True,
    behind_axes=[0, 1]):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    for box, color in zip(bboxes, colors):
        for pa, pb in zip(box, box[[1, 2, 3, 0]]):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha)
        if with_arrow:
            center = np.mean(box, axis=0)
            start = np.mean(np.concatenate([center[np.newaxis, ...], box[
                behind_axes]]), axis=0)
            front_axes = [i for i in range(4) if i not in behind_axes]
            end = np.mean(np.concatenate([center[np.newaxis, ...], box[
                front_axes]]), axis=0)
            ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[
                1], head_width=0.2, head_length=0.2, fc=color, ec=color)
    return ax
