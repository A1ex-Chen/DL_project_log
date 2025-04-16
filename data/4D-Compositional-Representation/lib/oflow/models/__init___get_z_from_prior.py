def get_z_from_prior(self, size=torch.Size([]), sample=True):
    """ Returns z from the prior distribution.

        If sample is true, z is sampled, otherwise the mean is returned.

        Args:
            size (torch.Size): size of z
            sample (bool): whether to sample z
        """
    if sample:
        z_t = self.p0_z.sample(size).to(self.device)
        z = self.p0_z.sample(size).to(self.device)
    else:
        z = self.p0_z.mean.to(self.device)
        z = z.expand(*size, *z.size())
        z_t = z
    return z, z_t
