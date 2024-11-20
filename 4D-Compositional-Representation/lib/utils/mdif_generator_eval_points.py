def eval_points(self, p, c_i, c_p=None, **kwargs):
    """ Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
    p_split = torch.split(p, self.points_batch_size)
    occ_hats = []
    for pi in p_split:
        pi = pi.unsqueeze(0).to(self.device)
        with torch.no_grad():
            occ_hat = self.model.decode_multiscale(pi, c_i, c_p, **kwargs
                ).logits
        occ_hats.append(occ_hat.squeeze(0).detach().cpu())
    occ_hat = torch.cat(occ_hats, dim=0)
    return occ_hat
