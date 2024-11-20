def to_mesh(self):
    occ = self.data
    nx, ny, nz = occ.shape
    grid_shape = nx + 1, ny + 1, nz + 1
    occ = np.pad(occ, 1, 'constant')
    f1_r = occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1]
    f2_r = occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1]
    f3_r = occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:]
    f1_l = ~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1]
    f2_l = ~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1]
    f3_l = ~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:]
    f1 = f1_r | f1_l
    f2 = f2_r | f2_l
    f3 = f3_r | f3_l
    assert f1.shape == (nx + 1, ny, nz)
    assert f2.shape == (nx, ny + 1, nz)
    assert f3.shape == (nx, ny, nz + 1)
    v = np.full(grid_shape, False)
    v[:, :-1, :-1] |= f1
    v[:, :-1, 1:] |= f1
    v[:, 1:, :-1] |= f1
    v[:, 1:, 1:] |= f1
    v[:-1, :, :-1] |= f2
    v[:-1, :, 1:] |= f2
    v[1:, :, :-1] |= f2
    v[1:, :, 1:] |= f2
    v[:-1, :-1, :] |= f3
    v[:-1, 1:, :] |= f3
    v[1:, :-1, :] |= f3
    v[1:, 1:, :] |= f3
    n_vertices = v.sum()
    v_idx = np.full(grid_shape, -1)
    v_idx[v] = np.arange(n_vertices)
    v_x, v_y, v_z = np.where(v)
    v_x = v_x / nx - 0.5
    v_y = v_y / ny - 0.5
    v_z = v_z / nz - 0.5
    vertices = np.stack([v_x, v_y, v_z], axis=1)
    f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
    f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
    f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)
    f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
    f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
    f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)
    faces_1_l = np.stack([v_idx[f1_l_x, f1_l_y, f1_l_z], v_idx[f1_l_x,
        f1_l_y, f1_l_z + 1], v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1], v_idx[
        f1_l_x, f1_l_y + 1, f1_l_z]], axis=1)
    faces_1_r = np.stack([v_idx[f1_r_x, f1_r_y, f1_r_z], v_idx[f1_r_x, 
        f1_r_y + 1, f1_r_z], v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1], v_idx[
        f1_r_x, f1_r_y, f1_r_z + 1]], axis=1)
    faces_2_l = np.stack([v_idx[f2_l_x, f2_l_y, f2_l_z], v_idx[f2_l_x + 1,
        f2_l_y, f2_l_z], v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1], v_idx[
        f2_l_x, f2_l_y, f2_l_z + 1]], axis=1)
    faces_2_r = np.stack([v_idx[f2_r_x, f2_r_y, f2_r_z], v_idx[f2_r_x,
        f2_r_y, f2_r_z + 1], v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1], v_idx[
        f2_r_x + 1, f2_r_y, f2_r_z]], axis=1)
    faces_3_l = np.stack([v_idx[f3_l_x, f3_l_y, f3_l_z], v_idx[f3_l_x, 
        f3_l_y + 1, f3_l_z], v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z], v_idx[
        f3_l_x + 1, f3_l_y, f3_l_z]], axis=1)
    faces_3_r = np.stack([v_idx[f3_r_x, f3_r_y, f3_r_z], v_idx[f3_r_x + 1,
        f3_r_y, f3_r_z], v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z], v_idx[
        f3_r_x, f3_r_y + 1, f3_r_z]], axis=1)
    faces = np.concatenate([faces_1_l, faces_1_r, faces_2_l, faces_2_r,
        faces_3_l, faces_3_r], axis=0)
    vertices = self.loc + self.scale * vertices
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh
