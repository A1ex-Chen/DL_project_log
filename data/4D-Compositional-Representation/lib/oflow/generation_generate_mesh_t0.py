def generate_mesh_t0(self, z=None, c_s=None, c_t=None, data=None, stats_dict={}
    ):
    """ Generates the mesh at time step t=0.

        Args:
            z (tensor): latent code z
            c_s (tensor): conditioned spatial code c_s
            c_t (tensor): conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): (time) statistics dictionary
        """
    if self.onet_generator.model.decoder is not None:
        mesh = self.onet_generator.generate_from_latent(z, c_s, stats_dict=
            stats_dict)
    else:
        vertices = data['mesh.vertices'][:, 0].squeeze(0).cpu().numpy()
        faces = data['mesh.faces'].squeeze(0).cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh
