def generate_motion_transfer(self, inp_id, inp_motion):
    self.onet_generator.model.eval()
    stats_dict = {}
    device = self.onet_generator.device
    inp_id = torch.from_numpy(inp_id['inputs']).unsqueeze(0).to(device)
    inp_motion = torch.from_numpy(inp_motion['inputs']).unsqueeze(0).to(device)
    meshes = []
    with torch.no_grad():
        t0 = time.time()
        c_s = self.onet_generator.model.encode_spatial_inputs(inp_id)
        c_t = self.onet_generator.model.encode_temporal_inputs(inp_motion)
        stats_dict['time (encode inputs)'] = time.time() - t0
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
