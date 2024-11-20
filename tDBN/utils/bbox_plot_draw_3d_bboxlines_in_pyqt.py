def draw_3d_bboxlines_in_pyqt(widget, bboxes, colors=GLColor.Green, width=
    1.0, labels=None, alpha=1.0, label_color='r', line_item=None,
    label_item=None):
    if bboxes.shape[0] == 0:
        bboxes = np.zeros([0, 8, 3])
    if not isinstance(colors, (list, np.ndarray)):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, (list, np.ndarray)):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = np.array([box[0], box[1], box[1], box[2], box[2], box[3],
            box[3], box[0], box[1], box[5], box[5], box[4], box[4], box[0],
            box[2], box[6], box[6], box[7], box[7], box[3], box[5], box[6],
            box[4], box[7]])
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    if bboxes.shape[0] != 0:
        total_lines = np.concatenate(total_lines, axis=0)
        total_colors = np.concatenate(total_colors, axis=0)
    else:
        total_lines = None
        total_colors = None
    if line_item is None:
        line_item = gl.GLLinePlotItem(pos=total_lines, color=total_colors,
            width=width, antialias=True, mode='lines')
        widget.addItem(line_item)
    else:
        line_item.setData(pos=total_lines, color=total_colors, width=width,
            antialias=True, mode='lines')
    label_color_qt = _pltcolor_to_qtcolor(label_color)
    if labels is not None:
        if label_item is None:
            label_item = GLLabelItem(bboxes[:, 0, :], labels, label_color_qt)
            label_item.setGLViewWidget(widget)
            widget.addItem(label_item)
        else:
            label_item.setData(pos=bboxes[:, 0, :], text=labels, color=
                label_color_qt)
    """
    for box, label in zip(bboxes, labels):
        if label is not None:
            label_color_qt = _pltcolor_to_qtcolor(label_color)
            t = GLTextItem(
                X=box[0, 0],
                Y=box[0, 1],
                Z=box[0, 2],
                text=label,
                color=label_color_qt)
            t.setGLViewWidget(widget)
            widget.addItem(t)
    """
    return line_item, label_item
