def draw_tracks(img, tracks, fill: (bool | None)=False, width: int=1, font:
    (str | None)=None, font_size: (int | None)=None):
    colors = []
    for track in tracks:
        color = tuple([(ord(c) * ord(c) % 256) for c in track.id[:3]])
        colors.append(color)
    bboxes = np.array([t.box for t in tracks])
    labels = [t.id[:3] for t in tracks]
    return draw_bounding_boxes(img, bboxes, labels, colors, fill, width,
        font, font_size)
