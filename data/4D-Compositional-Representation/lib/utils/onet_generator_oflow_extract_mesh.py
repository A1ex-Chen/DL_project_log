def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
    """ Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            z (tensor): latent code z
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + self.padding
    threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
    t0 = time.time()
    occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1000000.0)
    vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
    stats_dict['time (marching cubes)'] = time.time() - t0
    vertices -= 0.5
    vertices -= 1
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)
    if self.with_normals and not vertices.shape[0] == 0:
        t0 = time.time()
        normals = self.estimate_normals(vertices, z, c)
        stats_dict['time (normals)'] = time.time() - t0
    else:
        normals = None
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals,
        process=False)
    if vertices.shape[0] == 0:
        return mesh
    if self.simplify_nfaces is not None:
        t0 = time.time()
        mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.0)
        stats_dict['time (simplify)'] = time.time() - t0
    if self.refinement_step > 0:
        t0 = time.time()
        self.refine_mesh(mesh, occ_hat, z, c)
        stats_dict['time (refine)'] = time.time() - t0
    return mesh
