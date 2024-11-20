def generate_latent_space_interpolation(self, model_0, model_1,
    latent_space_file_path=None, n_samples=2, **kwargs):
    """ Generates a latent space interpolation.

        For usage, check generate_latent_space_interpolation.py.

        Args:
            model_0 (dict): dictionary for model_0
            model_1 (dict): dictionary for model_1
            latent_space_file_path (str): path to latent space file
            n_samples (int): how many samples to generate between start and end
        """
    self.onet_generator.model.eval()
    device = self.onet_generator.device
    assert self.fix_z != self.fix_zt
    df = pd.read_pickle(latent_space_file_path)
    k_interpolate = 'loc_z_t' if self.fix_z else 'loc_z'
    z0 = torch.from_numpy(df.loc[(df['model'] == model_0['model']) & (df[
        'start_idx'] == model_0['start_idx'])][k_interpolate].item()
        ).unsqueeze(0).to(device)
    zt = torch.from_numpy(df.loc[(df['model'] == model_1['model']) & (df[
        'start_idx'] == model_1['start_idx'])][k_interpolate].item()
        ).unsqueeze(0).to(device)
    k_fixed = 'loc_z' if self.fix_z else 'loc_z'
    fixed_z = torch.from_numpy(df.loc[(df['model'] == model_0['model']) & (
        df['start_idx'] == model_0['start_idx'])][k_fixed].item()).unsqueeze(0
        ).to(device)
    c_s = torch.empty(1, 0).to(device)
    c_t = torch.empty(1, 0).to(device)
    stats_dict = {}
    meshes = []
    with torch.no_grad():
        for i in range(n_samples):
            t0 = time.time()
            zi = zt * (i / (n_samples - 1)) + z0 * (1 - i / (n_samples - 1))
            if self.fix_z:
                z, z_t = fixed_z, zi
            else:
                z_t, z = fixed_z, zi
            stats_dict['time (encode inputs)'] = time.time() - t0
            mesh_t0 = self.generate_mesh_t0(z, c_s, c_t, None, stats_dict=
                stats_dict)
            meshes.append(mesh_t0)
            meshes_t = self.generate_meshes_t(mesh_t0.vertices, mesh_t0.
                faces, z=z_t, c_t=c_t, stats_dict=stats_dict)
            meshes.extend(meshes_t)
    return meshes, stats_dict
