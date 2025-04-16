def refine_mesh(self, mesh, occ_hat, z, c=None):
    """ Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
    self.model.eval()
    n_x, n_y, n_z = occ_hat.shape
    assert n_x == n_y == n_z
    threshold = self.threshold
    v0 = torch.FloatTensor(mesh.vertices).to(self.device)
    v = torch.nn.Parameter(v0.clone())
    faces = torch.LongTensor(mesh.faces).to(self.device)
    optimizer = optim.RMSprop([v], lr=0.0001)
    for it_r in trange(self.refinement_step):
        optimizer.zero_grad()
        face_vertex = v[faces]
        eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
        eps = torch.FloatTensor(eps).to(self.device)
        face_point = (face_vertex * eps[:, :, None]).sum(dim=1)
        face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
        face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
        face_normal = torch.cross(face_v1, face_v2)
        face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) +
            1e-10)
        face_value = torch.sigmoid(self.model.decode(face_point.unsqueeze(0
            ), z, c).logits)
        normal_target = -autograd.grad([face_value.sum()], [face_point],
            create_graph=True)[0]
        normal_target = normal_target / (normal_target.norm(dim=1, keepdim=
            True) + 1e-10)
        loss_target = (face_value - threshold).pow(2).mean()
        loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()
        loss = loss_target + 0.01 * loss_normal
        loss.backward()
        optimizer.step()
    mesh.vertices = v.data.cpu().numpy()
    return mesh
