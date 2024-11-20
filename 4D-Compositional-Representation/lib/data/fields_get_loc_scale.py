def get_loc_scale(self, mesh):
    """ Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        """
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max() / (1 - self.sample_padding)
    return loc, scale
