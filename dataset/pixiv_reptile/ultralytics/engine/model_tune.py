def tune(self, use_ray=False, iterations=10, *args, **kwargs):
    """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args (list): Variable length argument list for additional arguments.
            **kwargs (any): Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    self._check_is_pytorch_model()
    if use_ray:
        from ultralytics.utils.tuner import run_ray_tune
        return run_ray_tune(self, *args, max_samples=iterations, **kwargs)
    else:
        from .tuner import Tuner
        custom = {}
        args = {**self.overrides, **custom, **kwargs, 'mode': 'train'}
        return Tuner(args=args, _callbacks=self.callbacks)(model=self,
            iterations=iterations)
