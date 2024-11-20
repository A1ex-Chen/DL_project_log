def draw(self, frame, tracked_objects, coord_transform=None):
    frame_scale = frame.shape[0] / 100
    if self.radius is None:
        self.radius = int(max(frame_scale * 0.7, 1))
    if self.thickness is None:
        self.thickness = int(max(frame_scale / 7, 1))
    for obj in tracked_objects:
        if not obj.live_points.any():
            continue
        if self.color is None:
            color = Palette.choose_color(obj.id)
        else:
            color = self.color
        points_to_draw = self.get_points_to_draw(obj.get_estimate(absolute=
            True))
        for point in coord_transform.abs_to_rel(points_to_draw):
            Drawer.circle(frame, position=tuple(point.astype(int)), radius=
                self.radius, color=color, thickness=self.thickness)
        last = points_to_draw
        for i, past_points in enumerate(self.past_points[obj.id]):
            overlay = frame.copy()
            last = coord_transform.abs_to_rel(last)
            for j, point in enumerate(coord_transform.abs_to_rel(past_points)):
                Drawer.line(overlay, tuple(last[j].astype(int)), tuple(
                    point.astype(int)), color=color, thickness=self.thickness)
            last = past_points
            alpha = self.alphas[i]
            frame = Drawer.alpha_blend(overlay, frame, alpha=alpha)
        self.past_points[obj.id].insert(0, points_to_draw)
        self.past_points[obj.id] = self.past_points[obj.id][:self.max_history]
    return frame
