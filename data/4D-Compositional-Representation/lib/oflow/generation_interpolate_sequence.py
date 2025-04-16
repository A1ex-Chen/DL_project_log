def interpolate_sequence(self, data, n_time_steps=17, **kwargs):
    """ Generates an interpolation sequence.

        Args:
            data (dict): data dictionary
            mesh_dir (str): mesh directory
            modelname (str): model name
            n_time_steps (int): number of time steps which should be generated.
                If no value is passed, the standard object value for
                n_time_steps is used.
        """
    self.onet_generator.model.eval()
    device = self.onet_generator.device
    inputs_full = data.get('inputs', torch.empty(1, 1, 0)).to(device)
    faces = data['mesh.faces'].squeeze(0).cpu().numpy()
    vertices = data['mesh.vertices']
    num_files = inputs_full.shape[0]
    self.n_time_steps = n_time_steps
    meshes = []
    mesh_0 = trimesh.Trimesh(vertices=vertices[0].cpu().numpy(), faces=
        faces, process=False)
    meshes.append(mesh_0)
    for i in range(num_files - 1):
        inputs = inputs_full[i:i + 2].unsqueeze(0)
        vertex_data = data['mesh.vertices'][i:i + 2].unsqueeze(0)
        mesh_t0_vertices = vertices[i].cpu().numpy()
        with torch.no_grad():
            _, c_t = self.onet_generator.model.encode_inputs(inputs)
            z_t = self.onet_generator.model.get_z_from_prior((1,), sample=
                self.onet_generator.sample)[-1].to(device)
            meshes_t = self.generate_meshes_t(mesh_t0_vertices, faces, z=
                z_t, c_t=c_t, vertex_data=vertex_data)
            meshes.extend(meshes_t)
    return meshes
