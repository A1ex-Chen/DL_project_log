def transform_to_homogenous(self, p):
    device = self._device
    batch_size, num_points, _ = p.size()
    r = torch.sqrt(torch.sum(p ** 2, dim=2, keepdim=True))
    p_homo = torch.cat((p, torch.ones(batch_size, num_points, 1).to(device)
        ), dim=2) / r
    return p_homo
