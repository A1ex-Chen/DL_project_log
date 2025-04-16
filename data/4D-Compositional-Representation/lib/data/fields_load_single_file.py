def load_single_file(self, file_path):
    """ Loads a single file.

        Args:
            file_path (str): file path
        """
    pointcloud_dict = np.load(file_path)
    points = pointcloud_dict['points'].astype(np.float32)
    loc = pointcloud_dict['loc'].astype(np.float32)
    scale = pointcloud_dict['scale'].astype(np.float32)
    return points, loc, scale
