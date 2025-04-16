def predict(self, source: Union[str, Path, int, list, tuple, np.ndarray,
    torch.Tensor]=None, stream: bool=False, predictor=None, **kwargs) ->List[
    Results]:
    """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode. It also provides support for SAM-type models
        through 'prompts'.

        The method sets up a new predictor if not already present and updates its arguments with each call.
        It also issues a warning and uses default assets if the 'source' is not provided. The method determines if it
        is being called from the command line interface and adjusts its behavior accordingly, including setting defaults
        for confidence threshold and saving behavior.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor. Defaults to None.
            **kwargs (any): Additional keyword arguments for configuring the prediction process. These arguments allow
                for further customization of the prediction behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor is not properly set up.
        """
    if source is None:
        source = ASSETS
        LOGGER.warning(
            f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
    is_cli = (ARGV[0].endswith('yolo') or ARGV[0].endswith('ultralytics')
        ) and any(x in ARGV for x in ('predict', 'track', 'mode=predict',
        'mode=track'))
    custom = {'conf': 0.25, 'batch': 1, 'save': is_cli, 'mode': 'predict'}
    args = {**self.overrides, **custom, **kwargs}
    prompts = args.pop('prompts', None)
    if not self.predictor:
        self.predictor = predictor or self._smart_load('predictor')(overrides
            =args, _callbacks=self.callbacks)
        self.predictor.setup_model(model=self.model, verbose=is_cli)
    else:
        self.predictor.args = get_cfg(self.predictor.args, args)
        if 'project' in args or 'name' in args:
            self.predictor.save_dir = get_save_dir(self.predictor.args)
    if prompts and hasattr(self.predictor, 'set_prompts'):
        self.predictor.set_prompts(prompts)
    return self.predictor.predict_cli(source=source
        ) if is_cli else self.predictor(source=source, stream=stream)
