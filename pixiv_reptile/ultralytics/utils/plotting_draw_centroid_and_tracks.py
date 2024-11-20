def draw_centroid_and_tracks(self, track, color=(255, 0, 255),
    track_thickness=2):
    """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(self.im, [points], isClosed=False, color=color, thickness
        =track_thickness)
    cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), 
        track_thickness * 2, color, -1)
