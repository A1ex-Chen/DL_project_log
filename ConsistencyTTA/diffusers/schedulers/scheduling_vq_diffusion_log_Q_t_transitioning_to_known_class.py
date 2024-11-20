def log_Q_t_transitioning_to_known_class(self, *, t: torch.int, x_t: torch.
    LongTensor, log_onehot_x_t: torch.FloatTensor, cumulative: bool):
    """
        Returns the log probabilities of the rows from the (cumulative or non-cumulative) transition matrix for each
        latent pixel in `x_t`.

        See equation (7) for the complete non-cumulative transition matrix. The complete cumulative transition matrix
        is the same structure except the parameters (alpha, beta, gamma) are the cumulative analogs.

        Args:
            t (torch.Long):
                The timestep that determines which transition matrix is used.

            x_t (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`.

            log_onehot_x_t (`torch.FloatTensor` of shape `(batch size, num classes, num latent pixels)`):
                The log one-hot vectors of `x_t`

            cumulative (`bool`):
                If cumulative is `False`, we use the single step transition matrix `t-1`->`t`. If cumulative is `True`,
                we use the cumulative transition matrix `0`->`t`.

        Returns:
            `torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`:
                Each _column_ of the returned matrix is a _row_ of log probabilities of the complete probability
                transition matrix.

                When non cumulative, returns `self.num_classes - 1` rows because the initial latent pixel cannot be
                masked.

                Where:
                - `q_n` is the probability distribution for the forward process of the `n`th latent pixel.
                - C_0 is a class of a latent pixel embedding
                - C_k is the class of the masked latent pixel

                non-cumulative result (omitting logarithms):
                ```
                q_0(x_t | x_{t-1} = C_0) ... q_n(x_t | x_{t-1} = C_0)
                          .      .                     .
                          .               .            .
                          .                      .     .
                q_0(x_t | x_{t-1} = C_k) ... q_n(x_t | x_{t-1} = C_k)
                ```

                cumulative result (omitting logarithms):
                ```
                q_0_cumulative(x_t | x_0 = C_0)    ...  q_n_cumulative(x_t | x_0 = C_0)
                          .               .                          .
                          .                        .                 .
                          .                               .          .
                q_0_cumulative(x_t | x_0 = C_{k-1}) ... q_n_cumulative(x_t | x_0 = C_{k-1})
                ```
        """
    if cumulative:
        a = self.log_cumprod_at[t]
        b = self.log_cumprod_bt[t]
        c = self.log_cumprod_ct[t]
    else:
        a = self.log_at[t]
        b = self.log_bt[t]
        c = self.log_ct[t]
    if not cumulative:
        log_onehot_x_t_transitioning_from_masked = log_onehot_x_t[:, -1, :
            ].unsqueeze(1)
    log_onehot_x_t = log_onehot_x_t[:, :-1, :]
    log_Q_t = (log_onehot_x_t + a).logaddexp(b)
    mask_class_mask = x_t == self.mask_class
    mask_class_mask = mask_class_mask.unsqueeze(1).expand(-1, self.
        num_embed - 1, -1)
    log_Q_t[mask_class_mask] = c
    if not cumulative:
        log_Q_t = torch.cat((log_Q_t,
            log_onehot_x_t_transitioning_from_masked), dim=1)
    return log_Q_t
