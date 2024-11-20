def info(self, detailed=False, verbose=True):
    """
        Logs information about the SAM model.

        Args:
            detailed (bool, optional): If True, displays detailed information about the model. Defaults to False.
            verbose (bool, optional): If True, displays information on the console. Defaults to True.

        Returns:
            (tuple): A tuple containing the model's information.
        """
    return model_info(self.model, detailed=detailed, verbose=verbose)
