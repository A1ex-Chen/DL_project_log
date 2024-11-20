def normalize_mesh(self, mesh, loc, scale):
    """ Normalize mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        """
    mesh.apply_translation(-loc)
    mesh.apply_scale(1 / scale)
    return mesh
