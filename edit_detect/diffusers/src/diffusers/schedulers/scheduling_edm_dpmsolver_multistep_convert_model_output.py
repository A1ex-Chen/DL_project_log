def convert_model_output(self, model_output: torch.Tensor, sample: torch.
    Tensor=None) ->torch.Tensor:
    """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
    sigma = self.sigmas[self.step_index]
    x0_pred = self.precondition_outputs(sample, model_output, sigma)
    if self.config.thresholding:
        x0_pred = self._threshold_sample(x0_pred)
    return x0_pred
