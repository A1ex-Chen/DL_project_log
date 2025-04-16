def info(self, detailed: bool=False, verbose: bool=True):
    """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments passed.
        It can control the verbosity of the output.

        Args:
            detailed (bool): If True, shows detailed information about the model. Defaults to False.
            verbose (bool): If True, prints the information. If False, returns the information. Defaults to True.

        Returns:
            (list): Various types of information about the model, depending on the 'detailed' and 'verbose' parameters.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    self._check_is_pytorch_model()
    return self.model.info(detailed=detailed, verbose=verbose)
