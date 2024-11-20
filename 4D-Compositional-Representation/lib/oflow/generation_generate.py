def generate(self, data, return_stats=True, n_samples=1, **kwargs):
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
    inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
    vertex_data = data.get('mesh.vertices', None)
    meshes = []
    with torch.no_grad():
        for i in range(n_samples):
            t0 = time.time()
            c_s, c_t = self.onet_generator.model.encode_inputs(inputs)
            stats_dict['time (encode inputs)'] = time.time() - t0
            z, z_t = self.onet_generator.model.get_z_from_prior((1,),
                sample=self.onet_generator.sample)
            if self.fix_z:
                z = self.fixed_z
            if self.fix_zt:
                z_t = self.fixed_zt
            z = z.to(device)
            z_t = z_t.to(device)
            mesh_t0 = self.generate_mesh_t0(z, c_s, c_t, data, stats_dict=
                stats_dict)
            meshes.append(mesh_t0)
            meshes_t = self.generate_meshes_t(mesh_t0.vertices, mesh_t0.
                faces, z=z_t, c_t=c_t, vertex_data=vertex_data, stats_dict=
                stats_dict)
            meshes.extend(meshes_t)
    return meshes, stats_dict
