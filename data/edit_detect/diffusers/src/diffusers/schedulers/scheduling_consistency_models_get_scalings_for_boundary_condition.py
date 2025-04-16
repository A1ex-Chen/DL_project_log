def get_scalings_for_boundary_condition(self, sigma):
    """
        Gets the scalings used in the consistency model parameterization (from Appendix C of the
        [paper](https://huggingface.co/papers/2303.01469)) to enforce boundary condition.

        <Tip>

        `epsilon` in the equations for `c_skip` and `c_out` is set to `sigma_min`.

        </Tip>

        Args:
            sigma (`torch.Tensor`):
                The current sigma in the Karras sigma schedule.

        Returns:
            `tuple`:
                A two-element tuple where `c_skip` (which weights the current sample) is the first element and `c_out`
                (which weights the consistency model output) is the second element.
        """
    sigma_min = self.config.sigma_min
    sigma_data = self.config.sigma_data
    c_skip = sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
    c_out = (sigma - sigma_min) * sigma_data / (sigma ** 2 + sigma_data ** 2
        ) ** 0.5
    return c_skip, c_out
