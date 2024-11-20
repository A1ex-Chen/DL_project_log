def eval_velocity_field(self, t, p, z=None, c_t=None):
    """ Evaluates the velocity field at points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        """
    z_dim = z.shape[-1]
    c_dim = c_t.shape[-1]
    p = self.concat_vf_input(p, c=c_t, z=z)
    t_steps_eval = torch.tensor(0).float().to(t.device)
    out = self.vector_field(t_steps_eval, p, T_batch=t).unsqueeze(0)
    p_out = self.disentangle_vf_output(out, c_dim=c_dim, z_dim=z_dim,
        return_start=True)
    p_out = p_out.squeeze(1)
    return out
