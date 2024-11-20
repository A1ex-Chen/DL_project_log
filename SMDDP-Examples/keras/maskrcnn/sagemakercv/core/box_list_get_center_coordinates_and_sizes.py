def get_center_coordinates_and_sizes(self, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    box_corners = self.get()
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(a=box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.0
    xcenter = xmin + width / 2.0
    return [ycenter, xcenter, height, width]
