def draw_tracked_objects(frame: np.ndarray, objects: Sequence[
    'TrackedObject'], radius: Optional[int]=None, color: Optional[ColorLike
    ]=None, id_size: Optional[float]=None, id_thickness: Optional[int]=None,
    draw_points: bool=True, color_by_label: bool=False, draw_labels: bool=
    False, label_size: Optional[int]=None):
    """
    **Deprecated** use [`draw_points`][norfair.drawing.draw_points.draw_points]
    """
    warn_once('draw_tracked_objects is deprecated, use draw_points instead')
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))
    _draw_points_alias(frame=frame, drawables=objects, color='by_label' if
        color_by_label else color, radius=radius, thickness=None,
        draw_labels=draw_labels, draw_ids=id_size is not None and id_size >
        0, draw_points=draw_points, text_size=label_size or id_size,
        text_thickness=id_thickness, text_color=None, hide_dead_points=True)
