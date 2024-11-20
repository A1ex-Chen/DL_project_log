def _load(self, weights: str, task=None) ->None:
    """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
    if weights.lower().startswith(('https://', 'http://', 'rtsp://',
        'rtmp://', 'tcp://')):
        weights = checks.check_file(weights)
    weights = checks.check_model_file_from_stem(weights)
    if Path(weights).suffix == '.pt':
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = self.model.args['task']
        self.overrides = self.model.args = self._reset_ckpt_args(self.model
            .args)
        self.ckpt_path = self.model.pt_path
    else:
        weights = checks.check_file(weights)
        self.model, self.ckpt = weights, None
        self.task = task or guess_model_task(weights)
        self.ckpt_path = weights
    self.overrides['model'] = weights
    self.overrides['task'] = self.task
    self.model_name = weights
