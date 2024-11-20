def draw_3d_bbox_in_3dax(ax, bboxes, colors='r', alpha=0.25, facecolors=None):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(facecolors, list):
        facecolors = [facecolors for i in range(len(bboxes))]
    for box, color, facecolor in zip(bboxes, colors, facecolors):
        ax.scatter3D(box[:, 0], box[:, 1], box[:, 2], marker='.', color=color)
        verts = np.array([[box[0], box[1], box[2], box[3]], [box[4], box[5],
            box[6], box[7]], [box[0], box[3], box[7], box[4]], [box[1], box
            [2], box[6], box[5]], [box[0], box[1], box[5], box[4]], [box[3],
            box[2], box[6], box[7]]])
        mp3dcoll = Poly3DCollection(verts, linewidths=1, edgecolors=color,
            alpha=alpha, facecolors=facecolor)
        mp3dcoll.set_facecolor(facecolor)
        mp3dcoll.set_edgecolor(color)
        mp3dcoll.set_alpha(alpha)
        ax.add_collection3d(mp3dcoll)
    return ax
