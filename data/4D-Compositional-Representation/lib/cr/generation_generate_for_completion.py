def generate_for_completion(self, c_i, c_p, c_m):
    self.onet_generator.model.eval()
    stats_dict = {}
    meshes = []
    with torch.no_grad():
        c_s_at_t0 = torch.cat([c_i, c_p], 1)
        mesh_t0 = self.generate_mesh_t0(c_s_at_t0, stats_dict=stats_dict)
        meshes.append(mesh_t0)
        t = self.get_time_steps()
        c_p_at_t = self.onet_generator.model.transform_to_t_eval(t, p=c_p,
            c_t=c_m)
        c_s_at_t = torch.cat([c_i.unsqueeze(0).repeat(1, self.n_time_steps -
            1, 1), c_p_at_t], -1)
        meshes_t = self.generate_meshes_t(c_t=c_s_at_t, stats_dict=stats_dict)
        meshes.extend(meshes_t)
    return meshes, stats_dict
