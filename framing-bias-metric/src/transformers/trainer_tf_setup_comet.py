def setup_comet(self):
    """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE:
                (Optional): str - "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME:
                (Optional): str - Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY:
                (Optional): str - folder to use for saving offline experiments when `COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__
        """
    comet_mode = os.getenv('COMET_MODE', 'ONLINE').upper()
    args = {'project_name': os.getenv('COMET_PROJECT_NAME', 'huggingface')}
    experiment = None
    if comet_mode == 'ONLINE':
        experiment = comet_ml.Experiment(**args)
        logger.info('Automatic Comet.ml online logging enabled')
    elif comet_mode == 'OFFLINE':
        args['offline_directory'] = os.getenv('COMET_OFFLINE_DIRECTORY', './')
        experiment = comet_ml.OfflineExperiment(**args)
        logger.info(
            'Automatic Comet.ml offline logging enabled; use `comet upload` when finished'
            )
    if experiment is not None:
        experiment._set_model_graph(self.model, framework='transformers')
        experiment._log_parameters(self.args, prefix='args/', framework=
            'transformers')
        experiment._log_parameters(self.model.config, prefix='config/',
            framework='transformers')
