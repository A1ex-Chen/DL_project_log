def _load(self, weights: str, task=None):
    """
        Loads the specified weights into the SAM model.

        Args:
            weights (str): Path to the weights file.
            task (str, optional): Task name. Defaults to None.
        """
    self.model = build_sam(weights)
