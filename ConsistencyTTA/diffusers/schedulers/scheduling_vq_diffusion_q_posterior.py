def q_posterior(self, log_p_x_0, x_t, t):
    """
        Calculates the log probabilities for the predicted classes of the image at timestep `t-1`. I.e. Equation (11).

        Instead of directly computing equation (11), we use Equation (5) to restate Equation (11) in terms of only
        forward probabilities.

        Equation (11) stated in terms of forward probabilities via Equation (5):

        Where:
        - the sum is over x_0 = {C_0 ... C_{k-1}} (classes for x_0)

        p(x_{t-1} | x_t) = sum( q(x_t | x_{t-1}) * q(x_{t-1} | x_0) * p(x_0) / q(x_t | x_0) )

        Args:
            log_p_x_0: (`torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.

            x_t: (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`

            t (torch.Long):
                The timestep that determines which transition matrix is used.

        Returns:
            `torch.FloatTensor` of shape `(batch size, num classes, num latent pixels)`:
                The log probabilities for the predicted classes of the image at timestep `t-1`. I.e. Equation (11).
        """
    log_onehot_x_t = index_to_log_onehot(x_t, self.num_embed)
    log_q_x_t_given_x_0 = self.log_Q_t_transitioning_to_known_class(t=t,
        x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=True)
    log_q_t_given_x_t_min_1 = self.log_Q_t_transitioning_to_known_class(t=t,
        x_t=x_t, log_onehot_x_t=log_onehot_x_t, cumulative=False)
    q = log_p_x_0 - log_q_x_t_given_x_0
    q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
    q = q - q_log_sum_exp
    q = self.apply_cumulative_transitions(q, t - 1)
    log_p_x_t_min_1 = q + log_q_t_given_x_t_min_1 + q_log_sum_exp
    return log_p_x_t_min_1
