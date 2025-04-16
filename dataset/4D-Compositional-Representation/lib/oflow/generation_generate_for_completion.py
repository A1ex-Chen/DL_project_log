def generate_for_completion(self, c_s, c_t):
    """ Generates meshes for respective time steps.

        Args:
            data (dict): data dictionary
            mesh_dir (str): mesh directory
            modelname (str): model name
            return_stats (bool): whether to return (time) statistics
            n_samples (int): number of latent samples which should be used
        """
    self.onet_generator.model.eval()
    stats_dict = {}
    device = self.onet_generator.device
    meshes = []
    with torch.no_grad():
        z, z_t = self.onet_generator.model.get_z_from_prior((1,), sample=
            self.onet_generator.sample)
        if self.fix_z:
            z = self.fixed_z
        if self.fix_zt:
            z_t = self.fixed_zt
        z = z.to(device)
        z_t = z_t.to(device)
        mesh_t0 = self.generate_mesh_t0(z, c_s, c_t, stats_dict=stats_dict)
        meshes.append(mesh_t0)
        meshes_t = self.generate_meshes_t(mesh_t0.vertices, mesh_t0.faces,
            z=z_t, c_t=c_t, stats_dict=stats_dict)
        meshes.extend(meshes_t)
    return meshes, stats_dict
