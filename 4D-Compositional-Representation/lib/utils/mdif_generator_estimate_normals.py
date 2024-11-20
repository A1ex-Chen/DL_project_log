def estimate_normals(self, vertices, z, c=None):
    """ Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
    device = self.device
    vertices = torch.FloatTensor(vertices)
    vertices_split = torch.split(vertices, self.points_batch_size)
    normals = []
    z, c = z.unsqueeze(0), c.unsqueeze(0)
    for vi in vertices_split:
        vi = vi.unsqueeze(0).to(device)
        vi.requires_grad_()
        occ_hat = self.model.decode(vi, z, c).logits
        out = occ_hat.sum()
        out.backward()
        ni = -vi.grad
        ni = ni / torch.norm(ni, dim=-1, keepdim=True)
        ni = ni.squeeze(0).cpu().numpy()
        normals.append(ni)
    normals = np.concatenate(normals, axis=0)
    return normals
