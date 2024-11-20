def export_to_obj(mesh, output_obj_path: str=None):
    if output_obj_path is None:
        output_obj_path = tempfile.NamedTemporaryFile(suffix='.obj').name
    verts = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vertex_colors = np.stack([mesh.vertex_channels[x].detach().cpu().numpy(
        ) for x in 'RGB'], axis=1)
    vertices = ['{} {} {} {} {} {}'.format(*coord, *color) for coord, color in
        zip(verts.tolist(), vertex_colors.tolist())]
    faces = ['f {} {} {}'.format(str(tri[0] + 1), str(tri[1] + 1), str(tri[
        2] + 1)) for tri in faces.tolist()]
    combined_data = [('v ' + vertex) for vertex in vertices] + faces
    with open(output_obj_path, 'w') as f:
        f.writelines('\n'.join(combined_data))
