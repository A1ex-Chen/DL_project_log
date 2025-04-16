def forward(self, image):
    """Calculate the homography matrix for the given image.

        Parameters:
        - image: numpy array
            The source image.
        - dst_points: numpy array or None
            The destination points for the transformation. If not provided,
            it defaults to the four corners of a standard soccer pitch (105m x 68m).

        Returns:
        - numpy array
            The computed homography matrix.
        """
    contour = self._get_largest_contour(image)
    quadrilateral = self._approximate_quad(contour)
    homography_matrix = self._calculate_homography(quadrilateral, self.
        dst_points)
    return homography_matrix
