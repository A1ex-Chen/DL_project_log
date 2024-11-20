def query(self, points):
    point_indices, tri_indices = self.tri_hash.query(points)
    point_indices = np.array(point_indices, dtype=np.int64)
    tri_indices = np.array(tri_indices, dtype=np.int64)
    points = points[point_indices]
    triangles = self.triangles[tri_indices]
    mask = self.check_triangles(points, triangles)
    point_indices = point_indices[mask]
    tri_indices = tri_indices[mask]
    return point_indices, tri_indices
