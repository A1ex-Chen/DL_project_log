def pred_epsilon(self, sample, model_output, timestep):
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    if self.scheduler.config.prediction_type == 'epsilon':
        pred_eps = model_output
    elif self.scheduler.config.prediction_type == 'sample':
        pred_eps = (sample - alpha_prod_t ** 0.5 * model_output
            ) / beta_prod_t ** 0.5
    elif self.scheduler.config.prediction_type == 'v_prediction':
        pred_eps = (beta_prod_t ** 0.5 * sample + alpha_prod_t ** 0.5 *
            model_output)
    else:
        raise ValueError(
            f'prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
    return pred_eps
