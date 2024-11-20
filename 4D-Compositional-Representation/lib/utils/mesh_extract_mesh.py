def extract_mesh(self):
    threshold = self.threshold
    vertices = []
    triangles = []
    vertex_dict = dict()
    active_simplices = self.active_simplices()
    active_simplices.sort(axis=1)
    for simplex in active_simplices:
        new_vertices = []
        for i1, i2 in combinations(simplex, 2):
            assert i1 < i2
            v1 = self.values[i1]
            v2 = self.values[i2]
            if (v1 < threshold) ^ (v2 < threshold):
                vertex_idx = vertex_dict.get((i1, i2), len(vertices))
                vertex_idx = len(vertices)
                if vertex_idx == len(vertices):
                    tau = (threshold - v1) / (v2 - v1)
                    assert 0 <= tau <= 1
                    p = (1 - tau) * self.points[i1] + tau * self.points[i2]
                    vertices.append(p)
                    vertex_dict[i1, i2] = vertex_idx
                new_vertices.append(vertex_idx)
        assert len(new_vertices) in (3, 4)
        p0 = self.points[simplex[0]]
        v0 = self.values[simplex[0]]
        if len(new_vertices) == 3:
            i1, i2, i3 = new_vertices
            p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
            vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
            if vol * (v0 - threshold) <= 0:
                triangles.append((i1, i2, i3))
            else:
                triangles.append((i1, i3, i2))
        elif len(new_vertices) == 4:
            i1, i2, i3, i4 = new_vertices
            p1, p2, p3, p4 = vertices[i1], vertices[i2], vertices[i3
                ], vertices[i4]
            vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
            if vol * (v0 - threshold) <= 0:
                triangles.append((i1, i2, i3))
            else:
                triangles.append((i1, i3, i2))
            vol = get_tetrahedon_volume(np.asarray([p0, p2, p3, p4]))
            if vol * (v0 - threshold) <= 0:
                triangles.append((i2, i3, i4))
            else:
                triangles.append((i2, i4, i3))
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32)
    return vertices, triangles
