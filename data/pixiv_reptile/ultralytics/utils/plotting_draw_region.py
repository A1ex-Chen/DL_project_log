def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
    """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
    cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=
        True, color=color, thickness=thickness)
