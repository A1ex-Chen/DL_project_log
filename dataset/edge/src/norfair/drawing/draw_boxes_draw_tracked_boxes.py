def draw_tracked_boxes(frame: np.ndarray, objects: Sequence['TrackedObject'
    ], border_colors: Optional[Tuple[int, int, int]]=None, border_width:
    Optional[int]=None, id_size: Optional[int]=None, id_thickness: Optional
    [int]=None, draw_box: bool=True, color_by_label: bool=False,
    draw_labels: bool=False, label_size: Optional[int]=None, label_width:
    Optional[int]=None) ->np.array:
    """**Deprecated**. Use [`draw_box`][norfair.drawing.draw_boxes.draw_boxes]"""
    warn_once('draw_tracked_boxes is deprecated, use draw_box instead')
    return draw_boxes(frame=frame, drawables=objects, color='by_label' if
        color_by_label else border_colors, thickness=border_width,
        text_size=label_size or id_size, text_thickness=id_thickness or
        label_width, draw_labels=draw_labels, draw_ids=id_size is not None and
        id_size > 0, draw_box=draw_box)
