def draw_bbox_in_ax(ax, bboxes, rotations=None, fmt=FORMAT.Corner, labels=
    None, label_size='small', edgecolors='r', linestyle='dashed', alpha=0.5):
    if rotations is None:
        rotations = np.zeros([bboxes.shape[0]])
    else:
        rotations = rotations / np.pi * 180
    if labels is None:
        labels = [None] * bboxes.shape[0]
    if not isinstance(edgecolors, list):
        edgecolors = [edgecolors for i in range(len(bboxes))]
    if fmt == FORMAT.Corner:
        bboxes = corner_to_length(bboxes)
    for bbox, rot, e_color, label in zip(bboxes, rotations, edgecolors, labels
        ):
        rect_p = patches.Rectangle(bbox[:2], bbox[2], bbox[3], rot, fill=
            False, edgecolor=e_color, linestyle=linestyle, alpha=alpha)
        ax.add_patch(rect_p)
        if label is not None:
            t = ax.text(bbox[0], bbox[1], label, ha='left', va='bottom',
                color=e_color, size=label_size)
    return ax
