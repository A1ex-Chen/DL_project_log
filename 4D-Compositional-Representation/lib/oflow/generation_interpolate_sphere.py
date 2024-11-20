def interpolate_sphere(self, z1, z2, t):
    """ Interpolates on a sphere.

        Args:
            z1 (tensor): start latent code
            z2 (tensor): end latent code
            t (tensor): time steps
        """
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1 - t) * omega) / torch.sin(omega)
    s2 = torch.sin(t * omega) / torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z
