def find_quadrilateral(self, image):
    """Find the quadrilateral in the given image.

        Parameters:
        - image: numpy array
            The source image.

        Returns:
        - numpy array
            The quadrilateral in the image.
        """
    contour = self._get_largest_contour(image)
    hull = self._approximate_hull(contour)
    quadrilateral = self._approximate_quad(hull)
    return self.order_points(quadrilateral)
