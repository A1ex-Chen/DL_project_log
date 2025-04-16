def step_correct(self, model_output: torch.FloatTensor, sigma_hat: float,
    sigma_prev: float, sample_hat: torch.FloatTensor, sample_prev: torch.
    FloatTensor, derivative: torch.FloatTensor, return_dict: bool=True
    ) ->Union[KarrasVeOutput, Tuple]:
    """
        Correct the predicted sample based on the output model_output of the network. TODO complete description

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            sample_prev (`torch.FloatTensor`): TODO
            derivative (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class

        Returns:
            prev_sample (TODO): updated sample in the diffusion chain. derivative (TODO): TODO

        """
    pred_original_sample = sample_prev + sigma_prev * model_output
    derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
    sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative +
        0.5 * derivative_corr)
    if not return_dict:
        return sample_prev, derivative
    return KarrasVeOutput(prev_sample=sample_prev, derivative=derivative,
        pred_original_sample=pred_original_sample)
