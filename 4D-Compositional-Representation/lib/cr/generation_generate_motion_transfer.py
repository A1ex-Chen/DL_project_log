def generate_motion_transfer(self, inp_id, inp_motion):
    self.onet_generator.model.eval()
    stats_dict = {}
    device = self.onet_generator.device
    inp_id = torch.from_numpy(inp_id['inputs']).unsqueeze(0).to(device)
    inp_motion = torch.from_numpy(inp_motion['inputs']).unsqueeze(0).to(device)
    meshes = []
    with torch.no_grad():
        c_i = self.onet_generator.model.encoder_identity(inp_id[:, 0, :])
        c_p = self.onet_generator.model.encoder(inp_motion[:, 0, :])
        c_m = self.onet_generator.model.encoder_motion(inp_motion)
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
