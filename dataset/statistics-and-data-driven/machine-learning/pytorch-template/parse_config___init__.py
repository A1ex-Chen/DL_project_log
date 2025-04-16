def __init__(self, config, resume=None, modification=None, run_id=None):
    """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
    self._config = _update_config(config, modification)
    self.resume = resume
    save_dir = Path(self.config['trainer']['save_dir'])
    exper_name = self.config['name']
    if run_id is None:
        run_id = datetime.now().strftime('%m%d_%H%M%S')
    self._save_dir = save_dir / 'models' / exper_name / run_id
    self._log_dir = save_dir / 'log' / exper_name / run_id
    exist_ok = run_id == ''
    self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
    self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
    write_json(self.config, self.save_dir / 'config.json')
    setup_logging(self.log_dir)
    self.log_levels = {(0): logging.WARNING, (1): logging.INFO, (2):
        logging.DEBUG}
