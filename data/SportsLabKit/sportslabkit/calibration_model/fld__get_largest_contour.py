def _get_largest_contour(self, image):
    """Extract and return the largest contour from the binary image."""
    binary = self._preprocess_image(image)
    lines = self._get_lines(binary)
    line_image = np.zeros_like(binary)
    for line in lines:
        x0, y0, x1, y1 = map(int, line[0])
        cv2.line(line_image, (x0, y0), (x1, y1), 255, self.line_thickness)
    contours, _ = cv2.findContours(line_image, cv2.RETR_TREE, cv2.
        CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    return hull
