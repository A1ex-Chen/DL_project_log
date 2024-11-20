def __init__(self, triangles, resolution=128):
    self.triangles = triangles
    self.tri_hash = _TriangleHash(triangles, resolution)
