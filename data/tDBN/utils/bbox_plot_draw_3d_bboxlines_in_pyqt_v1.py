def draw_3d_bboxlines_in_pyqt_v1(widget, bboxes, colors=(0.0, 1.0, 0.0, 1.0
    ), width=1.0, labels=None, label_color='r'):
    if not isinstance(colors, list):
        colors = [colors for i in range(len(bboxes))]
    if not isinstance(labels, list):
        labels = [labels for i in range(len(bboxes))]
    for box, facecolor, label in zip(bboxes, colors, labels):
        lines = np.array([box[0], box[1], box[1], box[2], box[2], box[3],
            box[3], box[0], box[1], box[5], box[5], box[4], box[4], box[0],
            box[2], box[6], box[6], box[7], box[7], box[3], box[5], box[6],
            box[4], box[7]])
        color = np.array([list(facecolor) for i in range(len(lines))])
        plt = gl.GLLinePlotItem(pos=lines, color=color, width=width,
            antialias=True, mode='lines')
        widget.addItem(plt)
        if label is not None:
            label_color_qt = _pltcolor_to_qtcolor(label_color)
            t = GLTextItem(X=box[0, 0], Y=box[0, 1], Z=box[0, 2], text=
                label, color=label_color_qt)
            t.setGLViewWidget(widget)
            widget.addItem(t)
    return widget
