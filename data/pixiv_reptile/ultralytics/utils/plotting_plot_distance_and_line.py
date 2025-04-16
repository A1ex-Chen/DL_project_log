def plot_distance_and_line(self, distance_m, distance_mm, centroids,
    line_color, centroid_color):
    """
        Plot the distance and line on frame.

        Args:
            distance_m (float): Distance between two bbox centroids in meters.
            distance_mm (float): Distance between two bbox centroids in millimeters.
            centroids (list): Bounding box centroids data.
            line_color (RGB): Distance line color.
            centroid_color (RGB): Bounding box centroid color.
        """
    (text_width_m, text_height_m), _ = cv2.getTextSize(
        f'Distance M: {distance_m:.2f}m', 0, self.sf, self.tf)
    cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 10, 25 +
        text_height_m + 20), line_color, -1)
    cv2.putText(self.im, f'Distance M: {distance_m:.2f}m', (20, 50), 0,
        self.sf, centroid_color, self.tf, cv2.LINE_AA)
    (text_width_mm, text_height_mm), _ = cv2.getTextSize(
        f'Distance MM: {distance_mm:.2f}mm', 0, self.sf, self.tf)
    cv2.rectangle(self.im, (15, 75), (15 + text_width_mm + 10, 75 +
        text_height_mm + 20), line_color, -1)
    cv2.putText(self.im, f'Distance MM: {distance_mm:.2f}mm', (20, 100), 0,
        self.sf, centroid_color, self.tf, cv2.LINE_AA)
    cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
    cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
    cv2.circle(self.im, centroids[1], 6, centroid_color, -1)
