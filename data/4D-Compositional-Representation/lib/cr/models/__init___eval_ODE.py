def eval_ODE(self, t, p, c_t=None, t_batch=None, invert=False, return_start
    =False):
    """ Evaluates the ODE for points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            c_t (tensor): latent conditioned temporal code
            z (tensor): latent code
            t_batch (tensor): helper time tensor for batch processing of points
                with different time values when going backwards
            invert (bool): whether to invert the velocity field (used for
                batch processing of points with different time values)
            return_start (bool): whether to return the start points
        """
    c_dim = c_t.shape[-1]
    p_dim = p.shape[-1]
    t_steps_eval, t_order = self.return_time_steps(t)
    if len(t_steps_eval) == 1:
        return p.unsqueeze(1), t_order
    f_options = {'T_batch': t_batch, 'invert': invert}
    p = self.concat_vf_input(p, c=c_t)
    s = self.odeint(self.vector_field, p, t_steps_eval, method=self.
        ode_solver, rtol=self.rtol, atol=self.atol, options=self.
        ode_options, f_options=f_options)
    p_out = self.disentangle_vf_output(s, p_dim=p_dim, c_dim=c_dim,
        return_start=return_start)
    return p_out, t_order
