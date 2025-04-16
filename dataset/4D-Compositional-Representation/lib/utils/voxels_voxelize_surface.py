def voxelize_surface(mesh, resolution):
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = (vertices + 0.5) * resolution
    face_loc = vertices[faces]
    occ = np.full((resolution,) * 3, 0, dtype=np.int32)
    face_loc = face_loc.astype(np.float32)
    voxelize_mesh_(occ, face_loc)
    occ = occ != 0
    return occ
