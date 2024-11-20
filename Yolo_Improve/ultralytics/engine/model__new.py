def _new(self, cfg: str, task=None, model=None, verbose=False) ->None:
    """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
    cfg_dict = yaml_model_load(cfg)
    self.cfg = cfg
    self.task = task or guess_model_task(cfg_dict)
    self.model = (model or self._smart_load('model'))(cfg_dict, verbose=
        verbose and RANK == -1)
    self.overrides['model'] = self.cfg
    self.overrides['task'] = self.task
    self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
    self.model.task = self.task
    self.model_name = cfg
