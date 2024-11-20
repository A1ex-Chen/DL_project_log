def step(self, model_output: torch.FloatTensor, timestep: torch.long,
    sample: torch.LongTensor, generator: Optional[torch.Generator]=None,
    return_dict: bool=True) ->Union[VQDiffusionSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep via the reverse transition distribution i.e. Equation (11). See the
        docstring for `self.q_posterior` for more in depth docs on how Equation (11) is computed.

        Args:
            log_p_x_0: (`torch.FloatTensor` of shape `(batch size, num classes - 1, num latent pixels)`):
                The log probabilities for the predicted classes of the initial latent pixels. Does not include a
                prediction for the masked class as the initial unnoised image cannot be masked.

            t (`torch.long`):
                The timestep that determines which transition matrices are used.

            x_t: (`torch.LongTensor` of shape `(batch size, num latent pixels)`):
                The classes of each latent pixel at time `t`

            generator: (`torch.Generator` or None):
                RNG for the noise applied to p(x_{t-1} | x_t) before it is sampled from.

            return_dict (`bool`):
                option for returning tuple rather than VQDiffusionSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.VQDiffusionSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.VQDiffusionSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
    if timestep == 0:
        log_p_x_t_min_1 = model_output
    else:
        log_p_x_t_min_1 = self.q_posterior(model_output, sample, timestep)
    log_p_x_t_min_1 = gumbel_noised(log_p_x_t_min_1, generator)
    x_t_min_1 = log_p_x_t_min_1.argmax(dim=1)
    if not return_dict:
        return x_t_min_1,
    return VQDiffusionSchedulerOutput(prev_sample=x_t_min_1)
