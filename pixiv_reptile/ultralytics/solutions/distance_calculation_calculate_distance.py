def calculate_distance(self, centroid1, centroid2):
    """
        Calculates the distance between two centroids.

        Args:
            centroid1 (tuple): Coordinates of the first centroid (x, y).
            centroid2 (tuple): Coordinates of the second centroid (x, y).

        Returns:
            (tuple): Distance in meters and millimeters.
        """
    pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (
        centroid1[1] - centroid2[1]) ** 2)
    distance_m = pixel_distance / self.pixel_per_meter
    distance_mm = distance_m * 1000
    return distance_m, distance_mm
