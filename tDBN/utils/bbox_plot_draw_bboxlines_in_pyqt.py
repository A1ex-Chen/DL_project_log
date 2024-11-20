def draw_bboxlines_in_pyqt(widget, bboxes, colors=GLColor.Green, width=1.0,
    labels=None, alpha=1.0, label_color='r', line_item=None, label_item=None):
    if bboxes.shape[0] == 0:
        return
    if not isinstance(colors, list):
        if isinstance(colors, GLColor):
            colors = gl_color(colors, alpha)
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    total_lines = []
    total_colors = []
    for box, facecolor in zip(bboxes, colors):
        lines = np.array([box[0], box[1], box[1], box[2], box[2], box[3],
            box[3], box[0]])
        total_lines.append(lines)
        color = np.array([list(facecolor) for i in range(len(lines))])
        total_colors.append(color)
    total_lines = np.concatenate(total_lines, axis=0)
    total_colors = np.concatenate(total_colors, axis=0)
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
    return line_item, label_item
