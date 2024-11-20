def save_mesh_trimesh(save_path, verts, faces):
    mesh = trimesh.Trimesh(verts, faces, process=False)
    export_mesh(mesh, save_path)
