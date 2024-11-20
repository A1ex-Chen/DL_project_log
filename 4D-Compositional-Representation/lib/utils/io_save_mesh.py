def save_mesh(mesh, out_file, digits=10, face_colors=None):
    digits = int(digits)
    if face_colors is None:
        faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3, mesh
            .faces)).astype(np.int64)
    else:
        mesh.visual.face_colors = face_colors
        assert mesh.visual.face_colors.shape[0] == mesh.faces.shape[0]
        faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3, mesh
            .faces, mesh.visual.face_colors[:, :3])).astype(np.int64)
    export = 'OFF\n'
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += array_to_string(mesh.vertices, col_delim=' ', row_delim='\n',
        digits=digits) + '\n'
    export += array_to_string(faces_stacked, col_delim=' ', row_delim='\n')
    with open(out_file, 'w') as f:
        f.write(export)
    return mesh
