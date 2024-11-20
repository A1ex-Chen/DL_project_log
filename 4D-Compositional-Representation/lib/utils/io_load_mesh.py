def load_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        str_file = f.read().split('\n')
        n_vertices, n_faces, _ = list(map(lambda x: int(x), str_file[1].
            split(' ')))
        str_file = str_file[2:]
        v = [l.split(' ') for l in str_file[:n_vertices]]
        f = [l.split(' ') for l in str_file[n_vertices:]]
    v = np.array(v).astype(np.float32)
    f = np.array(f).astype(np.uint64)[:, 1:4]
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return mesh
