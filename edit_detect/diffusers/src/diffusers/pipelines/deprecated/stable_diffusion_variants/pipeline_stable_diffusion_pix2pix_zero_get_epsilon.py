def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor,
    timestep: int):
    pred_type = self.inverse_scheduler.config.prediction_type
    alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    if pred_type == 'epsilon':
        return model_output
    elif pred_type == 'sample':
        return (sample - alpha_prod_t ** 0.5 * model_output
            ) / beta_prod_t ** 0.5
    elif pred_type == 'v_prediction':
        return alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
    else:
        raise ValueError(
            f'prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
