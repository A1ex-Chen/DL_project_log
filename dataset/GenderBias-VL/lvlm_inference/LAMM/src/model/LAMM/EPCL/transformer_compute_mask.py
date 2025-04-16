def compute_mask(self, xyz, radius, dist=None):
    with torch.no_grad():
        if dist is None or dist.shape[1] != xyz.shape[1]:
            dist = torch.cdist(xyz, xyz, p=2)
        mask = dist >= radius
    return mask, dist
