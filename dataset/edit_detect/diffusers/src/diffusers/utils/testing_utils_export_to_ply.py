def export_to_ply(mesh, output_ply_path: str=None):
    """
    Write a PLY file for a mesh.
    """
    if output_ply_path is None:
        output_ply_path = tempfile.NamedTemporaryFile(suffix='.ply').name
    coords = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    rgb = np.stack([mesh.vertex_channels[x].detach().cpu().numpy() for x in
        'RGB'], axis=1)
    with buffered_writer(open(output_ply_path, 'wb')) as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(bytes(f'element vertex {len(coords)}\n', 'ascii'))
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        if rgb is not None:
            f.write(b'property uchar red\n')
            f.write(b'property uchar green\n')
            f.write(b'property uchar blue\n')
        if faces is not None:
            f.write(bytes(f'element face {len(faces)}\n', 'ascii'))
            f.write(b'property list uchar int vertex_index\n')
        f.write(b'end_header\n')
        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [(*coord, *rgb) for coord, rgb in zip(coords.tolist(
                ), rgb.tolist())]
            format = struct.Struct('<3f3B')
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct('<3f')
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))
        if faces is not None:
            format = struct.Struct('<B3I')
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))
    return output_ply_path
