def plot_box_and_track(self, track_id, box, cls, track):
    """
        Plots track and bounding box.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.
            cls (str): Object class name.
            track (list): Tracking history for drawing tracks path.
        """
    speed_label = (f'{int(self.dist_data[track_id])} km/h' if track_id in
        self.dist_data else self.names[int(cls)])
    bbox_color = colors(int(track_id)) if track_id in self.dist_data else (
        255, 0, 255)
    self.annotator.box_label(box, speed_label, bbox_color)
    cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 
        0), thickness=1)
    cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5,
        bbox_color, -1)
