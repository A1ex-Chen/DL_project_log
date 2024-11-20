def draw_bounding_box(widget, box_minmax, color):
    bbox = minmax_to_corner_3d(box_minmax)
    return draw_3d_bboxlines_in_pyqt(widget, bbox, color)
