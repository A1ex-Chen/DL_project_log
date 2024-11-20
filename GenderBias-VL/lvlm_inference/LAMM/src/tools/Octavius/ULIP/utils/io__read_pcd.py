@classmethod
def _read_pcd(cls, file_path):
    pc = open3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud
