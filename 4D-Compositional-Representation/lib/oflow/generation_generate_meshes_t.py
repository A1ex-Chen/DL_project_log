def generate_meshes_t(self, vertices_0, faces, z=None, c_t=None,
    vertex_data=None, stats_dict={}):
    """ Generates meshes for time steps t>0.

        Args:
            vertices_0 (numpy array): vertices of mesh at t=0
            faces (numpy array): faces of mesh at t=0
            z (tensor): latent code z
            c_t (tensor): temporal conditioned code c_t
            vertex_data (tensor): vertex tensor (start and end mesh if
                interpolation is required)
            stats_dict (dict): (time) statistics dictionary
        """
    device = self.onet_generator.device
    t = self.get_time_steps()
    vertices_0 = torch.from_numpy(vertices_0).to(device).unsqueeze(0).float()
    t0 = time.time()
    v_t_batch = self.onet_generator.model.transform_to_t(t, vertices_0, c_t
        =c_t, z=z)
    stats_dict['time (forward propagation)'] = time.time() - t0
    v_t_batch = v_t_batch.squeeze(0).cpu().numpy()
    if self.interpolate:
        vertices_t = vertex_data[:, -1].to(device)
        t0 = time.time()
        v_t_bw = self.onet_generator.model.transform_to_t_backward(t,
            vertices_t, c_t=c_t, z=z)
        stats_dict['time (backward propagation)'] = time.time() - t0
        v_t_bw = v_t_bw.squeeze(0).flip(0).cpu().numpy()[:-1]
        p_interpolate = self.return_interpolate(v_t_batch[:-1], v_t_bw)
        v_t_batch = np.concatenate([p_interpolate, vertices_t.cpu().numpy()])
    meshes = []
    for v_t in v_t_batch:
        meshes.append(trimesh.Trimesh(vertices=v_t, faces=faces, process=False)
            )
    return meshes
