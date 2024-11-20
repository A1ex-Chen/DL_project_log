@staticmethod
def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
    keep_ind = (points_2d[:, 0] > x1) * (points_2d[:, 1] > y1) * (points_2d
        [:, 0] < x2) * (points_2d[:, 1] < y2)
    return keep_ind
