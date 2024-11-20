def _get_epoch_indices(self, generator):
    """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
    rands = torch.rand(len(self._frac_part), generator=generator)
    rep_factors = self._int_part + (rands < self._frac_part).float()
    indices = []
    for dataset_index, rep_factor in enumerate(rep_factors):
        indices.extend([dataset_index] * int(rep_factor.item()))
    return torch.tensor(indices, dtype=torch.int64)
