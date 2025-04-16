def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([plydata['vertex']['x'], plydata['vertex']['y'],
        plydata['vertex']['z']], axis=1)
    return vertices
